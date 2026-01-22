from __future__ import annotations
import copy
import itertools
import json
import logging
import math
import os
import warnings
from functools import reduce
from math import gcd, isclose
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.fractions import lcm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, PeriodicSite, Structure, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list
from pymatgen.util.due import Doi, due
class ReconstructionGenerator:
    """Build a reconstructed Slab from a given initial Structure.

    This class needs a pre-defined dictionary specifying the parameters
    needed such as the SlabGenerator parameters, transformation matrix,
    sites to remove/add and slab/vacuum sizes.

    Attributes:
        slabgen_params (dict): Parameters for the SlabGenerator.
        trans_matrix (np.ndarray): A 3x3 transformation matrix to generate
            the reconstructed slab. Only the a and b lattice vectors are
            actually changed while the c vector remains the same.
            This matrix is what the Wood's notation is based on.
        reconstruction_json (dict): The full json or dictionary containing
            the instructions for building the slab.

    Todo:
        - Right now there is no way to specify what atom is being added.
            Use basis sets in the future?
    """

    def __init__(self, initial_structure: Structure, min_slab_size: float, min_vacuum_size: float, reconstruction_name: str) -> None:
        """Generates reconstructed slabs from a set of instructions.

        Args:
            initial_structure (Structure): Initial input structure. Note
                that to ensure that the Miller indices correspond to usual
                crystallographic definitions, you should supply a conventional
                unit cell structure.
            min_slab_size (float): Minimum Slab size in Angstrom.
            min_vacuum_size (float): Minimum vacuum layer size in Angstrom.
            reconstruction_name (str): Name of the dict containing the build
                instructions. The dictionary can contain any item, however
                any instructions archived in pymatgen for public use need
                to contain the following keys and items to ensure
                compatibility with the ReconstructionGenerator:

                    "name" (str): A descriptive name for the reconstruction,
                        typically including the type of structure,
                        the Miller index, the Wood's notation and additional
                        descriptors for the reconstruction.
                        Example: "fcc_110_missing_row_1x2"
                    "description" (str): A detailed description of the
                        reconstruction, intended to assist future contributors
                        in avoiding duplicate entries. Please read the description
                        carefully before adding to prevent duplications.
                    "reference" (str): Optional reference to the source of
                        the reconstruction.
                    "spacegroup" (dict): A dictionary indicating the space group
                        of the reconstruction. e.g. {"symbol": "Fm-3m", "number": 225}.
                    "miller_index" ([h, k, l]): Miller index of the reconstruction
                    "Woods_notation" (str): For a reconstruction, the a and b
                        lattice may change to accommodate the symmetry.
                        This notation indicates the change in
                        the vectors relative to the primitive (p) or
                        conventional (c) slab cell. E.g. p(2x1).

                        Reference: Wood, E. A. (1964). Vocabulary of surface
                        crystallography. Journal of Applied Physics, 35(4),
                        1306-1312.
                    "transformation_matrix" (numpy array): A 3x3 matrix to
                        transform the slab. Only the a and b lattice vectors
                        should change while the c vector remains the same.
                    "SlabGenerator_parameters" (dict): A dictionary containing
                        the parameters for the SlabGenerator, excluding the
                        miller_index, min_slab_size and min_vac_size. As the
                        Miller index is already specified and the min_slab_size
                        and min_vac_size can be changed regardless of the
                        reconstruction type. Having a consistent set of
                        SlabGenerator parameters allows for the instructions to
                        be reused.
                    "points_to_remove" (list[site]): A list of sites to
                        remove where the first two indices are fractional (in a
                        and b) and the third index is in units of 1/d (in c),
                        see the below "Notes" for details.
                    "points_to_add" (list[site]): A list of sites to add
                        where the first two indices are fractional (in a an b) and
                        the third index is in units of 1/d (in c), see the below
                        "Notes" for details.
                    "base_reconstruction" (dict, Optional): A dictionary specifying
                        an existing reconstruction model upon which the current
                        reconstruction is built to avoid repetition. E.g. the
                        alpha reconstruction of halites is based on the octopolar
                        reconstruction but with the topmost atom removed. The dictionary
                        for the alpha reconstruction would therefore contain the item
                        "reconstruction_base": "halite_111_octopolar_2x2", and
                        additional sites can be added by "points_to_add".

        Notes:
            1. For "points_to_remove" and "points_to_add", the third index
                for the c vector is specified in units of 1/d, where d represents
                the spacing between atoms along the hkl (the c vector), relative
                to the topmost site in the unreconstructed slab. For instance,
                a point of [0.5, 0.25, 1] corresponds to the 0.5 fractional
                coordinate of a, 0.25 fractional coordinate of b, and a
                distance of 1 atomic layer above the topmost site. Similarly,
                [0.5, 0.25, -0.5] corresponds to a point half an atomic layer
                below the topmost site, and [0.5, 0.25, 0] corresponds to a
                point at the same position along c as the topmost site.
                This approach is employed because while the primitive units
                of a and b remain constant, the user can vary the length
                of the c direction by adjusting the slab layer or the vacuum layer.

            2. The dictionary should only provide "points_to_remove" and
                "points_to_add" for the top surface. The ReconstructionGenerator
                will modify the bottom surface accordingly to return a symmetric Slab.
        """

        def build_recon_json() -> dict:
            """Build reconstruction instructions, optionally upon a base instruction set."""
            if reconstruction_name not in RECONSTRUCTIONS_ARCHIVE:
                raise KeyError(f"reconstruction_name={reconstruction_name!r} does not exist in the archive. Please select from one of the following: {list(RECONSTRUCTIONS_ARCHIVE)} or add it to the archive file 'reconstructions_archive.json'.")
            recon_json: dict = copy.deepcopy(RECONSTRUCTIONS_ARCHIVE[reconstruction_name])
            if 'base_reconstruction' in recon_json:
                new_points_to_add: list = []
                new_points_to_remove: list = []
                if 'points_to_add' in recon_json:
                    new_points_to_add = recon_json['points_to_add']
                if 'points_to_remove' in recon_json:
                    new_points_to_remove = recon_json['points_to_remove']
                recon_json = copy.deepcopy(RECONSTRUCTIONS_ARCHIVE[recon_json['base_reconstruction']])
                if 'points_to_add' in recon_json:
                    del recon_json['points_to_add']
                if new_points_to_add:
                    recon_json['points_to_add'] = new_points_to_add
                if 'points_to_remove' in recon_json:
                    del recon_json['points_to_remove']
                if new_points_to_remove:
                    recon_json['points_to_remove'] = new_points_to_remove
            return recon_json

        def build_slabgen_params() -> dict:
            """Build SlabGenerator parameters."""
            slabgen_params: dict = copy.deepcopy(recon_json['SlabGenerator_parameters'])
            slabgen_params['initial_structure'] = initial_structure.copy()
            slabgen_params['miller_index'] = recon_json['miller_index']
            slabgen_params['min_slab_size'] = min_slab_size
            slabgen_params['min_vacuum_size'] = min_vacuum_size
            return slabgen_params
        recon_json = build_recon_json()
        slabgen_params = build_slabgen_params()
        self.name = reconstruction_name
        self.slabgen_params = slabgen_params
        self.reconstruction_json = recon_json
        self.trans_matrix = recon_json['transformation_matrix']

    def build_slabs(self) -> list[Slab]:
        """Build reconstructed Slabs by:
            (1) Obtaining the unreconstructed Slab using the specified
                parameters for the SlabGenerator.
            (2) Applying the appropriate lattice transformation to the
                a and b lattice vectors.
            (3) Remove and then add specified sites from both surfaces.

        Returns:
            list[Slab]: The reconstructed slabs.
        """
        slabs = self.get_unreconstructed_slabs()
        recon_slabs = []
        for slab in slabs:
            z_spacing = get_d(slab)
            top_site = sorted(slab, key=lambda site: site.frac_coords[2])[-1].coords
            if 'points_to_remove' in self.reconstruction_json:
                sites_to_rm: list = copy.deepcopy(self.reconstruction_json['points_to_remove'])
                for site in sites_to_rm:
                    site[2] = slab.lattice.get_fractional_coords([top_site[0], top_site[1], top_site[2] + site[2] * z_spacing])[2]
                    cart_point = slab.lattice.get_cartesian_coords(site)
                    distances: list[float] = [site.distance_from_point(cart_point) for site in slab]
                    nearest_site = distances.index(min(distances))
                    slab.symmetrically_remove_atoms(indices=[nearest_site])
            if 'points_to_add' in self.reconstruction_json:
                sites_to_add: list = copy.deepcopy(self.reconstruction_json['points_to_add'])
                for site in sites_to_add:
                    site[2] = slab.lattice.get_fractional_coords([top_site[0], top_site[1], top_site[2] + site[2] * z_spacing])[2]
                    slab.symmetrically_add_atom(species=slab[0].specie, point=site)
            slab.reconstruction = self.name
            slab.recon_trans_matrix = self.trans_matrix
            ouc = slab.oriented_unit_cell.copy()
            ouc.make_supercell(self.trans_matrix)
            slab.oriented_unit_cell = ouc
            recon_slabs.append(slab)
        return recon_slabs

    def get_unreconstructed_slabs(self) -> list[Slab]:
        """Generate the unreconstructed (super) Slabs.

        TODO (@DanielYang59): this should be a private method.
        """
        return [slab.make_supercell(self.trans_matrix) for slab in SlabGenerator(**self.slabgen_params).get_slabs()]