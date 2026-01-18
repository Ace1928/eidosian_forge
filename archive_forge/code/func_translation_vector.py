from __future__ import annotations
import re
import string
import typing
import warnings
from math import cos, pi, sin, sqrt
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.electronic_structure.core import Magmom
from pymatgen.util.due import Doi, due
from pymatgen.util.string import transformation_to_string
@property
def translation_vector(self) -> np.ndarray:
    """A rank 1 numpy.array of dim 3 representing the translation vector."""
    return self.affine_matrix[0:3][:, 3]