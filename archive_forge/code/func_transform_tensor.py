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
def transform_tensor(self, tensor: np.ndarray) -> np.ndarray:
    """Applies rotation portion to a tensor. Note that tensor has to be in
        full form, not the Voigt form.

        Args:
            tensor (numpy array): A rank n tensor

        Returns:
            Transformed tensor.
        """
    dim = tensor.shape
    rank = len(dim)
    assert all((val == 3 for val in dim))
    lc = string.ascii_lowercase
    indices = (lc[:rank], lc[rank:2 * rank])
    einsum_string = ','.join((a + i for a, i in zip(*indices)))
    einsum_string += f',{indices[::-1][0]}->{indices[::-1][1]}'
    einsum_args = [self.rotation_matrix] * rank + [tensor]
    return np.einsum(einsum_string, *einsum_args)