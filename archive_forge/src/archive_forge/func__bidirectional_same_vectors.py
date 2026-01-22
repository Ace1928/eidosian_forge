from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util.due import Doi, due
from pymatgen.util.numba import njit
@njit
def _bidirectional_same_vectors(vec_set1, vec_set2, max_length_tol, max_angle_tol):
    """Bidirectional version of above matching constraint check."""
    return _unidirectional_is_same_vectors(vec_set1, vec_set2, max_length_tol=max_length_tol, max_angle_tol=max_angle_tol) or _unidirectional_is_same_vectors(vec_set2, vec_set1, max_length_tol=max_length_tol, max_angle_tol=max_angle_tol)