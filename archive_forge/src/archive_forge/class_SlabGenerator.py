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
class SlabGenerator:
    """Generate different slabs using shift values determined by where
    a unique termination can be found, along with other criteria such as where a
    termination doesn't break a polyhedral bond. The shift value then indicates
    where the slab layer will begin and terminate in the slab-vacuum system.

    Attributes:
        oriented_unit_cell (Structure): An oriented unit cell of the parent structure.
        parent (Structure): Parent structure from which Slab was derived.
        lll_reduce (bool): Whether the slabs will be orthogonalized.
        center_slab (bool): Whether the slabs will be centered in the slab-vacuum system.
        slab_scale_factor (float): Scale factor that brings
            the parent cell to the surface cell.
        miller_index (tuple): Miller index of plane parallel to surface.
        min_slab_size (float): Minimum size of layers containing atoms, in angstroms.
        min_vac_size (float): Minimum vacuum layer size, in angstroms.
    """

    def __init__(self, initial_structure: Structure, miller_index: tuple[int, int, int], min_slab_size: float, min_vacuum_size: float, lll_reduce: bool=False, center_slab: bool=False, in_unit_planes: bool=False, primitive: bool=True, max_normal_search: int | None=None, reorient_lattice: bool=True) -> None:
        """Calculates the slab scale factor and uses it to generate an
        oriented unit cell (OUC) of the initial structure.
        Also stores the initial information needed later on to generate a slab.

        Args:
            initial_structure (Structure): Initial input structure. Note that to
                ensure that the Miller indices correspond to usual
                crystallographic definitions, you should supply a conventional
                unit cell structure.
            miller_index ([h, k, l]): Miller index of the plane parallel to
                the surface. Note that this is referenced to the input structure.
                If you need this to be based on the conventional cell,
                you should supply the conventional structure.
            min_slab_size (float): In Angstroms or number of hkl planes
            min_vacuum_size (float): In Angstroms or number of hkl planes
            lll_reduce (bool): Whether to perform an LLL reduction on the
                final structure.
            center_slab (bool): Whether to center the slab in the cell with
                equal vacuum spacing from the top and bottom.
            in_unit_planes (bool): Whether to set min_slab_size and min_vac_size
                in number of hkl planes or Angstrom (default).
                Setting in units of planes is useful to ensure some slabs
                to have a certain number of layers, e.g. for Cs(100), 10 Ang
                will result in a slab with only 2 layers, whereas
                Fe(100) will have more layers. The slab thickness
                will be in min_slab_size/math.ceil(self._proj_height/dhkl)
                multiples of oriented unit cells.
            primitive (bool): Whether to reduce generated slabs to
                primitive cell. Note this does NOT generate a slab
                from a primitive cell, it means that after slab
                generation, we attempt to reduce the generated slab to
                primitive cell.
            max_normal_search (int): If set to a positive integer, the code
                will search for a normal lattice vector that is as
                perpendicular to the surface as possible, by considering
                multiple linear combinations of lattice vectors up to
                this value. This has no bearing on surface energies,
                but may be useful as a preliminary step to generate slabs
                for absorption or other sizes. It may not be the smallest possible
                cell for simulation. Normality is not guaranteed, but the oriented
                cell will have the c vector as normal as possible to the surface.
                The max absolute Miller index is usually sufficient.
            reorient_lattice (bool): reorient the lattice such that
                the c direction is parallel to the third lattice vector
        """

        def reduce_vector(vector: tuple[int, int, int]) -> tuple[int, int, int]:
            """Helper function to reduce vectors."""
            divisor = abs(reduce(gcd, vector))
            return cast(tuple[int, int, int], tuple((int(idx / divisor) for idx in vector)))

        def add_site_types() -> None:
            """Add Wyckoff symbols and equivalent sites to the initial structure."""
            if 'bulk_wyckoff' not in initial_structure.site_properties or 'bulk_equivalent' not in initial_structure.site_properties:
                spg_analyzer = SpacegroupAnalyzer(initial_structure)
                initial_structure.add_site_property('bulk_wyckoff', spg_analyzer.get_symmetry_dataset()['wyckoffs'])
                initial_structure.add_site_property('bulk_equivalent', spg_analyzer.get_symmetry_dataset()['equivalent_atoms'].tolist())

        def calculate_surface_normal() -> np.ndarray:
            """Calculate the unit surface normal vector using the reciprocal
            lattice vector.
            """
            recip_lattice = lattice.reciprocal_lattice_crystallographic
            normal = recip_lattice.get_cartesian_coords(miller_index)
            normal /= np.linalg.norm(normal)
            return normal

        def calculate_scaling_factor() -> np.ndarray:
            """Calculate scaling factor.

            # TODO (@DanielYang59): revise docstring to add more details.
            """
            slab_scale_factor = []
            non_orth_ind = []
            eye = np.eye(3, dtype=int)
            for idx, miller_idx in enumerate(miller_index):
                if miller_idx == 0:
                    slab_scale_factor.append(eye[idx])
                else:
                    d = abs(np.dot(normal, lattice.matrix[idx])) / lattice.abc[idx]
                    non_orth_ind.append((idx, d))
            c_index, _dist = max(non_orth_ind, key=lambda t: t[1])
            if len(non_orth_ind) > 1:
                lcm_miller = lcm(*(miller_index[i] for i, _d in non_orth_ind))
                for (ii, _di), (jj, _dj) in itertools.combinations(non_orth_ind, 2):
                    scale_factor = [0, 0, 0]
                    scale_factor[ii] = -int(round(lcm_miller / miller_index[ii]))
                    scale_factor[jj] = int(round(lcm_miller / miller_index[jj]))
                    slab_scale_factor.append(scale_factor)
                    if len(slab_scale_factor) == 2:
                        break
            if max_normal_search is None:
                slab_scale_factor.append(eye[c_index])
            else:
                index_range = sorted(range(-max_normal_search, max_normal_search + 1), key=lambda x: -abs(x))
                candidates = []
                for uvw in itertools.product(index_range, index_range, index_range):
                    if not any(uvw) or abs(np.linalg.det([*slab_scale_factor, uvw])) < 1e-08:
                        continue
                    vec = lattice.get_cartesian_coords(uvw)
                    osdm = np.linalg.norm(vec)
                    cosine = abs(np.dot(vec, normal) / osdm)
                    candidates.append((uvw, cosine, osdm))
                    if isclose(abs(cosine), 1, abs_tol=1e-08):
                        break
                uvw, cosine, osdm = max(candidates, key=lambda x: (x[1], -x[2]))
                slab_scale_factor.append(uvw)
            slab_scale_factor = np.array(slab_scale_factor)
            if np.linalg.det(slab_scale_factor) < 0:
                slab_scale_factor *= -1
            reduced_scale_factor = [reduce_vector(v) for v in slab_scale_factor]
            return np.array(reduced_scale_factor)
        add_site_types()
        lattice = initial_structure.lattice
        miller_index = reduce_vector(miller_index)
        normal = calculate_surface_normal()
        slab_scale_factor = calculate_scaling_factor()
        single = initial_structure.copy()
        single.make_supercell(slab_scale_factor)
        self.oriented_unit_cell = Structure.from_sites(single, to_unit_cell=True)
        self.max_normal_search = max_normal_search
        self.parent = initial_structure
        self.lll_reduce = lll_reduce
        self.center_slab = center_slab
        self.slab_scale_factor = slab_scale_factor
        self.miller_index = miller_index
        self.min_vac_size = min_vacuum_size
        self.min_slab_size = min_slab_size
        self.in_unit_planes = in_unit_planes
        self.primitive = primitive
        self._normal = normal
        self.reorient_lattice = reorient_lattice
        _a, _b, c = self.oriented_unit_cell.lattice.matrix
        self._proj_height = abs(np.dot(normal, c))

    def get_slab(self, shift: float=0, tol: float=0.1, energy: float | None=None) -> Slab:
        """Generate a slab based on a given shift value along the lattice c direction.

        Note:
            You should rarely use this (private) method directly, which is
            intended for other generation methods.

        Args:
            shift (float): The shift value along the lattice c direction in Angstrom.
            tol (float): Tolerance to determine primitive cell.
            energy (float): The energy to assign to the slab.

        Returns:
            Slab: from a shifted oriented unit cell.
        """
        scale_factor = self.slab_scale_factor
        height = self._proj_height
        height_per_layer = round(height / self.parent.lattice.d_hkl(self.miller_index), 8)
        if self.in_unit_planes:
            n_layers_slab = math.ceil(self.min_slab_size / height_per_layer)
            n_layers_vac = math.ceil(self.min_vac_size / height_per_layer)
        else:
            n_layers_slab = math.ceil(self.min_slab_size / height)
            n_layers_vac = math.ceil(self.min_vac_size / height)
        n_layers = n_layers_slab + n_layers_vac
        a, b, c = self.oriented_unit_cell.lattice.matrix
        new_lattice = [a, b, n_layers * c]
        species = self.oriented_unit_cell.species_and_occu
        frac_coords = self.oriented_unit_cell.frac_coords
        frac_coords = np.array(frac_coords) + np.array([0, 0, -shift])[None, :]
        frac_coords -= np.floor(frac_coords)
        frac_coords[:, 2] = frac_coords[:, 2] / n_layers
        all_coords = []
        for idx in range(n_layers_slab):
            _frac_coords = frac_coords.copy()
            _frac_coords[:, 2] += idx / n_layers
            all_coords.extend(_frac_coords)
        props = self.oriented_unit_cell.site_properties
        props = {k: v * n_layers_slab for k, v in props.items()}
        slab = Structure(new_lattice, species * n_layers_slab, all_coords, site_properties=props)
        if self.lll_reduce:
            lll_slab = slab.copy(sanitize=True)
            slab = lll_slab
            mapping = lll_slab.lattice.find_mapping(slab.lattice)
            if mapping is None:
                raise RuntimeError('LLL reduction has failed')
            scale_factor = np.dot(mapping[2], scale_factor)
        if self.center_slab:
            c_center = np.average([coord[2] for coord in slab.frac_coords])
            slab.translate_sites(list(range(len(slab))), [0, 0, 0.5 - c_center])
        if self.primitive:
            prim_slab = slab.get_primitive_structure(tolerance=tol)
            slab = prim_slab
            if energy is not None:
                energy *= prim_slab.volume / slab.volume
        ouc = self.oriented_unit_cell.copy()
        if self.primitive:
            slab_l = slab.lattice
            ouc = ouc.get_primitive_structure(constrain_latt={'a': slab_l.a, 'b': slab_l.b, 'alpha': slab_l.alpha, 'beta': slab_l.beta, 'gamma': slab_l.gamma})
            ouc = ouc if slab_l.a == ouc.lattice.a and slab_l.b == ouc.lattice.b else self.oriented_unit_cell
        return Slab(slab.lattice, slab.species_and_occu, slab.frac_coords, self.miller_index, ouc, shift, scale_factor, reorient_lattice=self.reorient_lattice, site_properties=slab.site_properties, energy=energy)

    def get_slabs(self, bonds: dict[tuple[Species | Element, Species | Element], float] | None=None, ftol: float=0.1, tol: float=0.1, max_broken_bonds: int=0, symmetrize: bool=False, repair: bool=False) -> list[Slab]:
        """Generate slabs with shift values calculated from the internal
        calculate_possible_shifts method. If the user decide to avoid breaking
        any polyhedral bond (by setting `bonds`), any shift value that do so
        would be filtered out.

        Args:
            bonds (dict): A {(species1, species2): max_bond_dist} dict.
                For example, PO4 groups may be defined as {("P", "O"): 3}.
            tol (float): Fractional tolerance for getting primitive cells
                and matching structures.
            ftol (float): Threshold for fcluster to check if two atoms are
                on the same plane. Default to 0.1 Angstrom in the direction of
                the surface normal.
            max_broken_bonds (int): Maximum number of allowable broken bonds
                for the slab. Use this to limit number of slabs. Defaults to 0,
                which means no bonds could be broken.
            symmetrize (bool): Whether to enforce the equivalency of slab surfaces.
            repair (bool): Whether to repair terminations with broken bonds (True)
                or just omit them (False). Default to False as repairing terminations
                can lead to many more possible slabs.

        Returns:
            list[Slab]: All possible Slabs of a particular surface,
                sorted by the number of bonds broken.
        """

        def gen_possible_shifts(ftol: float) -> list[float]:
            """Generate possible shifts by clustering z coordinates.

            Args:
                ftol (float): Threshold for fcluster to check if
                    two atoms are on the same plane.
            """
            frac_coords = self.oriented_unit_cell.frac_coords
            n_atoms = len(frac_coords)
            if n_atoms == 1:
                shift = frac_coords[0][2] + 0.5
                return [shift - math.floor(shift)]
            dist_matrix = np.zeros((n_atoms, n_atoms))
            for i, j in itertools.combinations(list(range(n_atoms)), 2):
                if i != j:
                    z_dist = frac_coords[i][2] - frac_coords[j][2]
                    z_dist = abs(z_dist - round(z_dist)) * self._proj_height
                    dist_matrix[i, j] = z_dist
                    dist_matrix[j, i] = z_dist
            z_matrix = linkage(squareform(dist_matrix))
            clusters = fcluster(z_matrix, ftol, criterion='distance')
            clst_loc = {c: frac_coords[i][2] for i, c in enumerate(clusters)}
            possible_clst = [coord - math.floor(coord) for coord in sorted(clst_loc.values())]
            n_shifts = len(possible_clst)
            shifts = []
            for i in range(n_shifts):
                if i == n_shifts - 1:
                    shift = (possible_clst[0] + 1 + possible_clst[i]) * 0.5
                else:
                    shift = (possible_clst[i] + possible_clst[i + 1]) * 0.5
                shifts.append(shift - math.floor(shift))
            return sorted(shifts)

        def get_z_ranges(bonds: dict[tuple[Species | Element, Species | Element], float]) -> list[tuple[float, float]]:
            """Collect occupied z ranges where each z_range is a (lower_z, upper_z) tuple.

            This method examines all sites in the oriented unit cell (OUC)
            and considers all neighboring sites within the specified bond distance
            for each site. If a site and its neighbor meet bonding and species
            requirements, their respective z-ranges will be collected.

            Args:
                bonds (dict): A {(species1, species2): max_bond_dist} dict.
                tol (float): Fractional tolerance for determine overlapping positions.
            """
            bonds = {(get_el_sp(s1), get_el_sp(s2)): dist for (s1, s2), dist in bonds.items()}
            z_ranges = []
            for (sp1, sp2), bond_dist in bonds.items():
                for site in self.oriented_unit_cell:
                    if sp1 in site.species:
                        for nn in self.oriented_unit_cell.get_neighbors(site, bond_dist):
                            if sp2 in nn.species:
                                z_range = tuple(sorted([site.frac_coords[2], nn.frac_coords[2]]))
                                if z_range[1] > 1:
                                    z_ranges.extend([(z_range[0], 1), (0, z_range[1] - 1)])
                                elif z_range[0] < 0:
                                    z_ranges.extend([(0, z_range[1]), (z_range[0] + 1, 1)])
                                elif z_range[0] != z_range[1]:
                                    z_ranges.append(z_range)
            return z_ranges
        z_ranges = [] if bonds is None else get_z_ranges(bonds)
        slabs = []
        for shift in gen_possible_shifts(ftol=ftol):
            bonds_broken = 0
            for z_range in z_ranges:
                if z_range[0] <= shift <= z_range[1]:
                    bonds_broken += 1
            slab = self.get_slab(shift=shift, tol=tol, energy=bonds_broken)
            if bonds_broken <= max_broken_bonds:
                slabs.append(slab)
            elif repair and bonds is not None:
                slabs.append(self.repair_broken_bonds(slab=slab, bonds=bonds))
        matcher = StructureMatcher(ltol=tol, stol=tol, primitive_cell=False, scale=False)
        final_slabs = []
        for group in matcher.group_structures(slabs):
            if symmetrize:
                sym_slabs = self.nonstoichiometric_symmetrized_slab(group[0])
                final_slabs.extend(sym_slabs)
            else:
                final_slabs.append(group[0])
        if symmetrize:
            matcher_sym = StructureMatcher(ltol=tol, stol=tol, primitive_cell=False, scale=False)
            final_slabs = [group[0] for group in matcher_sym.group_structures(final_slabs)]
        return sorted(final_slabs, key=lambda slab: slab.energy)

    def repair_broken_bonds(self, slab: Slab, bonds: dict[tuple[Species | Element, Species | Element], float]) -> Slab:
        """Repair broken bonds (specified by the bonds parameter) due to
        slab cleaving, and repair them by moving undercoordinated atoms
        to the other surface.

        How it works:
            For example a P-O4 bond may have P and O(4-x) on one side
            of the surface, and Ox on the other side, this method would
            first move P (the reference atom) to the other side,
            find its missing nearest neighbours (Ox), and move P
            and Ox back together.

        Args:
            slab (Slab): The Slab to repair.
            bonds (dict): A {(species1, species2): max_bond_dist} dict.
                For example, PO4 groups may be defined as {("P", "O"): 3}.

        Returns:
            Slab: The repaired Slab.
        """
        for species_pair, bond_dist in bonds.items():
            cn_dict = {}
            for idx, ele in enumerate(species_pair):
                cn_list = []
                for site in self.oriented_unit_cell:
                    ref_cn = 0
                    if site.species_string == ele:
                        for nn in self.oriented_unit_cell.get_neighbors(site, bond_dist):
                            if nn[0].species_string == species_pair[idx - 1]:
                                ref_cn += 1
                    cn_list.append(ref_cn)
                cn_dict[ele] = cn_list
            if max(cn_dict[species_pair[0]]) > max(cn_dict[species_pair[1]]):
                ele_ref, ele_other = species_pair
            else:
                ele_other, ele_ref = species_pair
            for idx, site in enumerate(slab):
                if site.species_string == ele_ref:
                    ref_cn = sum((1 if neighbor.species_string == ele_other else 0 for neighbor in slab.get_neighbors(site, bond_dist)))
                    if ref_cn not in cn_dict[ele_ref]:
                        slab = self.move_to_other_side(slab, [idx])
                        neighbors = slab.get_neighbors(slab[idx], r=bond_dist)
                        to_move = [nn[2] for nn in neighbors if nn[0].species_string == ele_other]
                        to_move.append(idx)
                        slab = self.move_to_other_side(slab, to_move)
        return slab

    def move_to_other_side(self, init_slab: Slab, index_of_sites: list[int]) -> Slab:
        """Move surface sites to the opposite surface of the Slab.

        If a selected site resides on the top half of the Slab,
        it would be moved to the bottom side, and vice versa.
        The distance moved is equal to the thickness of the Slab.

        Note:
            You should only use this method on sites close to the
            surface, otherwise it would end up deep inside the
            vacuum layer.

        Args:
            init_slab (Slab): The Slab whose sites would be moved.
            index_of_sites (list[int]): Indices representing
                the sites to move.

        Returns:
            Slab: The Slab with selected sites moved.
        """
        height: float = self._proj_height
        if self.in_unit_planes:
            height /= self.parent.lattice.d_hkl(self.miller_index)
        n_layers_slab: int = math.ceil(self.min_slab_size / height)
        n_layers_vac: int = math.ceil(self.min_vac_size / height)
        n_layers: int = n_layers_slab + n_layers_vac
        frac_dist: float = n_layers_slab / n_layers
        top_site_index: list[int] = []
        bottom_site_index: list[int] = []
        for idx in index_of_sites:
            if init_slab[idx].frac_coords[2] >= init_slab.center_of_mass[2]:
                top_site_index.append(idx)
            else:
                bottom_site_index.append(idx)
        slab = init_slab.copy()
        slab.translate_sites(top_site_index, vector=[0, 0, -frac_dist], frac_coords=True)
        slab.translate_sites(bottom_site_index, vector=[0, 0, frac_dist], frac_coords=True)
        return Slab(init_slab.lattice, slab.species, slab.frac_coords, init_slab.miller_index, init_slab.oriented_unit_cell, init_slab.shift, init_slab.scale_factor, energy=init_slab.energy)

    def nonstoichiometric_symmetrized_slab(self, init_slab: Slab) -> list[Slab]:
        """Symmetrize the two surfaces of a Slab, but may break the stoichiometry.

        How it works:
            1. Check whether two surfaces of the slab are equivalent.
            If the point group of the slab has an inversion symmetry (
            ie. belong to one of the Laue groups), then it's assumed that the
            surfaces are equivalent.

            2.If not symmetrical, sites at the bottom of the slab will be removed
            until the slab is symmetric, which may break the stoichiometry.

        Args:
            init_slab (Slab): The initial Slab.

        Returns:
            list[Slabs]: The symmetrized Slabs.
        """
        if init_slab.is_symmetric():
            return [init_slab]
        non_stoich_slabs = []
        for surface in ('top', 'bottom'):
            is_sym: bool = False
            slab = init_slab.copy()
            slab.energy = init_slab.energy
            while not is_sym:
                z_coords: list[float] = [site[2] for site in slab.frac_coords]
                if surface == 'top':
                    slab.remove_sites([z_coords.index(max(z_coords))])
                else:
                    slab.remove_sites([z_coords.index(min(z_coords))])
                if len(slab) <= len(self.parent):
                    warnings.warn('Too many sites removed, please use a larger slab.')
                    break
                if slab.is_symmetric():
                    is_sym = True
                    non_stoich_slabs.append(slab)
        return non_stoich_slabs