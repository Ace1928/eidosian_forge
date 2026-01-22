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
def _get_key_line_orbits(self, key_points, key_lines, key_points_inds_orbits):
    key_lines_copy = dict(zip(range(len(key_lines)), key_lines))
    key_lines_inds_orbits = []
    i = 0
    while len(key_lines_copy) > 0:
        key_lines_inds_orbits.append([])
        l0ind = next(iter(key_lines_copy))
        l0 = key_lines_copy[l0ind]
        key_lines_inds_orbits[i].append(l0)
        key_lines_copy.pop(l0ind)
        to_pop = []
        p00 = key_points[l0[0]]
        p01 = key_points[l0[1]]
        pmid0 = p00 + e / pi * (p01 - p00)
        for ind_key in key_lines_copy:
            l1 = key_lines_copy[ind_key]
            p10 = key_points[l1[0]]
            p11 = key_points[l1[1]]
            equivptspar = False
            equivptsperp = False
            equivline = False
            if np.array([l0[0] in orbit and l1[0] in orbit for orbit in key_points_inds_orbits]).any() and np.array([l0[1] in orbit and l1[1] in orbit for orbit in key_points_inds_orbits]).any():
                equivptspar = True
            elif np.array([l0[1] in orbit and l1[0] in orbit for orbit in key_points_inds_orbits]).any() and np.array([l0[0] in orbit and l1[1] in orbit for orbit in key_points_inds_orbits]).any():
                equivptsperp = True
            if equivptspar:
                pmid1 = p10 + e / pi * (p11 - p10)
                for op in self._rpg:
                    if not equivline:
                        p00pr = np.dot(op, p00)
                        diff0 = p10 - p00pr
                        if self._all_ints(diff0, atol=self._atol):
                            pmid0pr = np.dot(op, pmid0) + diff0
                            p01pr = np.dot(op, p01) + diff0
                            if np.allclose(p11, p01pr, atol=self._atol) and np.allclose(pmid1, pmid0pr, atol=self._atol):
                                equivline = True
            elif equivptsperp:
                pmid1 = p11 + e / pi * (p10 - p11)
                for op in self._rpg:
                    if not equivline:
                        p00pr = np.dot(op, p00)
                        diff0 = p11 - p00pr
                        if self._all_ints(diff0, atol=self._atol):
                            pmid0pr = np.dot(op, pmid0) + diff0
                            p01pr = np.dot(op, p01) + diff0
                            if np.allclose(p10, p01pr, atol=self._atol) and np.allclose(pmid1, pmid0pr, atol=self._atol):
                                equivline = True
            if equivline:
                key_lines_inds_orbits[i].append(l1)
                to_pop.append(ind_key)
        for key in to_pop:
            key_lines_copy.pop(key)
        i += 1
    return key_lines_inds_orbits