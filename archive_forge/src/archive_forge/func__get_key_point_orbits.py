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
def _get_key_point_orbits(self, key_points):
    key_points_copy = dict(zip(range(len(key_points) - 1), key_points[0:len(key_points) - 1]))
    key_points_inds_orbits = []
    i = 0
    while len(key_points_copy) > 0:
        key_points_inds_orbits.append([])
        k0ind = next(iter(key_points_copy))
        k0 = key_points_copy[k0ind]
        key_points_inds_orbits[i].append(k0ind)
        key_points_copy.pop(k0ind)
        for op in self._rpg:
            to_pop = []
            k1 = np.dot(op, k0)
            for ind_key in key_points_copy:
                diff = k1 - key_points_copy[ind_key]
                if self._all_ints(diff, atol=self._atol):
                    key_points_inds_orbits[i].append(ind_key)
                    to_pop.append(ind_key)
            for key in to_pop:
                key_points_copy.pop(key)
        i += 1
    key_points_inds_orbits.append([len(key_points) - 1])
    return key_points_inds_orbits