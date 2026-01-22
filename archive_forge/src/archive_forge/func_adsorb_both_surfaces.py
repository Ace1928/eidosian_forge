from __future__ import annotations
import itertools
import os
from typing import TYPE_CHECKING
import numpy as np
from matplotlib import patches
from matplotlib.path import Path
from monty.serialization import loadfn
from scipy.spatial import Delaunay
from pymatgen import vis
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Molecule, Structure
from pymatgen.core.operations import SymmOp
from pymatgen.core.surface import generate_all_slabs
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list_pbc
def adsorb_both_surfaces(self, molecule, repeat=None, min_lw=5.0, translate=True, reorient=True, find_args=None):
    """Function that generates all adsorption structures for a given
        molecular adsorbate on both surfaces of a slab. This is useful for
        calculating surface energy where both surfaces need to be equivalent or
        if we want to calculate nonpolar systems.

        Args:
            molecule (Molecule): molecule corresponding to adsorbate
            repeat (3-tuple or list): repeat argument for supercell generation
            min_lw (float): minimum length and width of the slab, only used
                if repeat is None
            reorient (bool): flag on whether or not to reorient adsorbate
                along the miller index
            find_args (dict): dictionary of arguments to be passed to the
                call to self.find_adsorption_sites, e.g. {"distance":2.0}
        """
    find_args = find_args or {}
    ad_slabs = self.generate_adsorption_structures(molecule, repeat=repeat, min_lw=min_lw, translate=translate, reorient=reorient, find_args=find_args)
    new_ad_slabs = []
    for ad_slab in ad_slabs:
        _, adsorbates, indices = (False, [], [])
        for idx, site in enumerate(ad_slab):
            if site.surface_properties == 'adsorbate':
                adsorbates.append(site)
                indices.append(idx)
        ad_slab.remove_sites(indices)
        slab = ad_slab.copy()
        for adsorbate in adsorbates:
            p2 = ad_slab.get_symmetric_site(adsorbate.frac_coords)
            slab.append(adsorbate.specie, p2, properties={'surface_properties': 'adsorbate'})
            slab.append(adsorbate.specie, adsorbate.frac_coords, properties={'surface_properties': 'adsorbate'})
        new_ad_slabs.append(slab)
    return new_ad_slabs