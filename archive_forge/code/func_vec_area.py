from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util.due import Doi, due
from pymatgen.util.numba import njit
@njit
def vec_area(a, b):
    """Area of lattice plane defined by two vectors."""
    return fast_norm(np.cross(a, b))