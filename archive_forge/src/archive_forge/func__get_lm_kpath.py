from __future__ import annotations
import itertools
from warnings import warn
import networkx as nx
import numpy as np
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.core import Spin
from pymatgen.symmetry.analyzer import cite_conventional_cell_algo
from pymatgen.symmetry.kpath import KPathBase, KPathLatimerMunro, KPathSeek, KPathSetyawanCurtarolo
def _get_lm_kpath(self, has_magmoms, magmom_axis, symprec, angle_tolerance, atol):
    """
        Returns:
            Latimer and Munro k-path with labels.
        """
    return KPathLatimerMunro(self._structure, has_magmoms, magmom_axis, symprec, angle_tolerance, atol)