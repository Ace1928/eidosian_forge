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
def as_xyz_str(self) -> str:
    """Returns a string of the form 'x, y, z', '-x, -y, z', '-y+1/2, x+1/2, z+1/2', etc.
        Only works for integer rotation matrices.
        """
    if not np.all(np.isclose(self.rotation_matrix, np.round(self.rotation_matrix))):
        warnings.warn('Rotation matrix should be integer')
    return transformation_to_string(self.rotation_matrix, translation_vec=self.translation_vector, delim=', ')