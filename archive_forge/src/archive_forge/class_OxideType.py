from __future__ import annotations
import collections
import itertools
from math import acos, pi
from typing import TYPE_CHECKING
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from pymatgen.analysis.local_env import JmolNN, VoronoiNN
from pymatgen.core import Composition, Element, PeriodicSite, Species
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class OxideType:
    """Separate class for determining oxide type."""

    def __init__(self, structure: Structure, relative_cutoff=1.1):
        """
        Args:
            structure: Input structure.
            relative_cutoff: Relative_cutoff * act. cutoff stipulates the max.
                distance two O atoms must be from each other. Default value is
                1.1. At most 1.1 is recommended, nothing larger, otherwise the
                script cannot distinguish between superoxides and peroxides.
        """
        self.structure = structure
        self.relative_cutoff = relative_cutoff
        self.oxide_type, self.nbonds = self.parse_oxide()

    def parse_oxide(self) -> tuple[str, int]:
        """
        Determines if an oxide is a peroxide/superoxide/ozonide/normal oxide.

        Returns:
            oxide_type (str): Type of oxide
            ozonide/peroxide/superoxide/hydroxide/None.
            nbonds (int): Number of peroxide/superoxide/hydroxide bonds in structure.
        """
        structure = self.structure
        relative_cutoff = self.relative_cutoff
        o_sites_frac_coords = []
        h_sites_frac_coords = []
        lattice = structure.lattice
        if isinstance(structure.elements[0], Element):
            comp = structure.composition
        elif isinstance(structure.elements[0], Species):
            elem_map: dict[Element, float] = collections.defaultdict(float)
            for site in structure:
                for species, occu in site.species.items():
                    elem_map[species.element] += occu
            comp = Composition(elem_map)
        if Element('O') not in comp or comp.is_element:
            return ('None', 0)
        for site in structure:
            syms = [sp.symbol for sp in site.species]
            if 'O' in syms:
                o_sites_frac_coords.append(site.frac_coords)
            if 'H' in syms:
                h_sites_frac_coords.append(site.frac_coords)
        if h_sites_frac_coords:
            dist_matrix = lattice.get_all_distances(o_sites_frac_coords, h_sites_frac_coords)
            if np.any(dist_matrix < relative_cutoff * 0.93):
                return ('hydroxide', int(len(np.where(dist_matrix < relative_cutoff * 0.93)[0]) / 2))
        dist_matrix = lattice.get_all_distances(o_sites_frac_coords, o_sites_frac_coords)
        np.fill_diagonal(dist_matrix, 1000)
        is_superoxide = False
        is_peroxide = False
        is_ozonide = False
        if np.any(dist_matrix < relative_cutoff * 1.35):
            bond_atoms = np.where(dist_matrix < relative_cutoff * 1.35)[0]
            is_superoxide = True
        elif np.any(dist_matrix < relative_cutoff * 1.49):
            is_peroxide = True
            bond_atoms = np.where(dist_matrix < relative_cutoff * 1.49)[0]
        if is_superoxide and len(bond_atoms) > len(set(bond_atoms)):
            is_superoxide = False
            is_ozonide = True
        try:
            n_bonds = len(set(bond_atoms))
        except UnboundLocalError:
            n_bonds = 0
        if is_ozonide:
            str_oxide = 'ozonide'
        elif is_superoxide:
            str_oxide = 'superoxide'
        elif is_peroxide:
            str_oxide = 'peroxide'
        else:
            str_oxide = 'oxide'
        if str_oxide == 'oxide':
            n_bonds = int(comp['O'])
        return (str_oxide, n_bonds)