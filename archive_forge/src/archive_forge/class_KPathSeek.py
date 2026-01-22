from __future__ import annotations
import abc
import itertools
from math import ceil, cos, e, pi, sin, tan
from typing import TYPE_CHECKING, Any
from warnings import warn
import networkx as nx
import numpy as np
import spglib
from monty.dev import requires
from scipy.linalg import sqrtm
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import MagSymmOp, SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, cite_conventional_cell_algo
class KPathSeek(KPathBase):
    """This class looks for a path along high-symmetry lines in the Brillouin zone. It is based on
    Hinuma, Y., Pizzi, G., Kumagai, Y., Oba, F., & Tanaka, I. (2017). Band structure diagram paths
    based on crystallography. Computational Materials Science, 128, 140-184.
    https://doi.org/10.1016/j.commatsci.2016.10.015. It should be used with primitive structures that
    comply with the definition given in the paper. The symmetry is determined by spglib using the
    SpacegroupAnalyzer class. k-points are generated using the get_kpoints() method for the
    reciprocal cell basis defined in the paper.
    """

    @requires(get_path is not None, 'SeeK-path needs to be installed to use the convention of Hinuma et al. (2015)')
    def __init__(self, structure: Structure, symprec: float=0.01, angle_tolerance=5, atol=1e-05, system_is_tri=True):
        """
        Args:
            structure (Structure): Structure object
            symprec (float): Tolerance for symmetry finding
            angle_tolerance (float): Angle tolerance for symmetry finding.
            atol (float): Absolute tolerance used to determine edge cases
                for settings of structures.
            system_is_tri (bool): Indicates if the system is time-reversal
                invariant.
        """
        super().__init__(structure, symprec=symprec, angle_tolerance=angle_tolerance, atol=atol)
        positions = structure.frac_coords
        sp = structure.site_properties
        species = [site.species for site in structure]
        site_data = species
        if not system_is_tri:
            warn("Non-zero 'magmom' data will be used to define unique atoms in the cell.")
            site_data = zip(species, [tuple(vec) for vec in sp['magmom']])
        unique_species: list[SpeciesLike] = []
        numbers = []
        for species, group in itertools.groupby(site_data):
            if species in unique_species:
                ind = unique_species.index(species)
                numbers.extend([ind + 1] * len(tuple(group)))
            else:
                unique_species.append(species)
                numbers.extend([len(unique_species)] * len(tuple(group)))
        cell = (self._latt.matrix, positions, numbers)
        lattice, scale_pos, atom_num = spglib.standardize_cell(cell, to_primitive=False, no_idealize=True, symprec=symprec)
        spg_struct = (lattice, scale_pos, atom_num)
        spath_dat = get_path(spg_struct, system_is_tri, 'hpkot', atol, symprec, angle_tolerance)
        self._tmat = self._trans_sc_to_Hin(spath_dat['bravais_lattice_extended'])
        self._rec_lattice = Lattice(spath_dat['reciprocal_primitive_lattice'])
        spath_data_formatted = [[spath_dat['path'][0][0]]]
        count = 0
        for pnum in range(len(spath_dat['path']) - 1):
            if spath_dat['path'][pnum][1] == spath_dat['path'][pnum + 1][0]:
                spath_data_formatted[count].append(spath_dat['path'][pnum][1])
            else:
                spath_data_formatted[count].append(spath_dat['path'][pnum][1])
                spath_data_formatted.append([])
                count += 1
                spath_data_formatted[count].append(spath_dat['path'][pnum + 1][0])
        spath_data_formatted[-1].append(spath_dat['path'][-1][1])
        self._kpath = {'kpoints': spath_dat['point_coords'], 'path': spath_data_formatted}

    @staticmethod
    def _trans_sc_to_Hin(sub_class):
        if sub_class in ['cP1', 'cP2', 'cF1', 'cF2', 'cI1', 'tP1', 'oP1', 'hP1', 'hP2', 'tI1', 'tI2', 'oF1', 'oF3', 'oI1', 'oI3', 'oC1', 'hR1', 'hR2', 'aP1', 'aP2', 'aP3', 'oA1']:
            return np.eye(3)
        if sub_class == 'oF2':
            return np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        if sub_class == 'oI2':
            return np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        if sub_class == 'oI3':
            return np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        if sub_class == 'oA2':
            return np.diag((-1, 1, -1))
        if sub_class == 'oC2':
            return np.diag((-1, 1, -1))
        if sub_class in ['mP1', 'mC1', 'mC2', 'mC3']:
            return np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        raise RuntimeError('Sub-classification of crystal not found!')