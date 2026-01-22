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
def _get_orbit_labels(self, orbit_cosines_orig, key_points_inds_orbits, atol):
    orbit_cosines_copy = orbit_cosines_orig.copy()
    orbit_labels_unsorted = [(len(key_points_inds_orbits) - 1, 26)]
    orbit_inds_remaining = range(len(key_points_inds_orbits) - 1)
    pop_orbits = []
    pop_labels = []
    for i, orb_cos in enumerate(orbit_cosines_copy):
        if np.isclose(orb_cos[0][1], 1.0, atol=atol):
            orbit_labels_unsorted.append((i, orb_cos[0][0]))
            pop_orbits.append(i)
            pop_labels.append(orb_cos[0][0])
    orbit_cosines_copy = self._reduce_cosines_array(orbit_cosines_copy, pop_orbits, pop_labels)
    orbit_inds_remaining = [i for i in orbit_inds_remaining if i not in pop_orbits]
    while len(orbit_labels_unsorted) < len(orbit_cosines_orig) + 1:
        pop_orbits = []
        pop_labels = []
        max_cosine_value = max((orb_cos[0][1] for orb_cos in orbit_cosines_copy))
        max_cosine_value_inds = [j for j in range(len(orbit_cosines_copy)) if orbit_cosines_copy[j][0][1] == max_cosine_value]
        max_cosine_label_inds = self._get_max_cosine_labels([orbit_cosines_copy[j] for j in max_cosine_value_inds], key_points_inds_orbits, atol)
        for j, label_ind in enumerate(max_cosine_label_inds):
            orbit_labels_unsorted.append((orbit_inds_remaining[max_cosine_value_inds[j]], label_ind))
            pop_orbits.append(max_cosine_value_inds[j])
            pop_labels.append(label_ind)
        orbit_cosines_copy = self._reduce_cosines_array(orbit_cosines_copy, pop_orbits, pop_labels)
        orbit_inds_remaining = [orbit_inds_remaining[j] for j in range(len(orbit_inds_remaining)) if j not in pop_orbits]
    orbit_labels = np.zeros(len(key_points_inds_orbits))
    for tup in orbit_labels_unsorted:
        orbit_labels[tup[0]] = tup[1]
    return orbit_labels