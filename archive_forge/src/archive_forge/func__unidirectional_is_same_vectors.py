from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util.due import Doi, due
from pymatgen.util.numba import njit
@njit
def _unidirectional_is_same_vectors(vec_set1, vec_set2, max_length_tol, max_angle_tol):
    """
    Determine if two sets of vectors are the same within length and angle
    tolerances
    Args:
        vec_set1(array[array]): an array of two vectors
        vec_set2(array[array]): second array of two vectors.
    """
    if np.absolute(rel_strain(vec_set1[0], vec_set2[0])) > max_length_tol:
        return False
    if np.absolute(rel_strain(vec_set1[1], vec_set2[1])) > max_length_tol:
        return False
    if np.absolute(rel_angle(vec_set1, vec_set2)) > max_angle_tol:
        return False
    return True