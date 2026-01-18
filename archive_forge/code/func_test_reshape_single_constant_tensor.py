from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
@staticmethod
def test_reshape_single_constant_tensor(scipy_backend):
    a = sp.csc_matrix(np.tile(np.arange(6), 3).reshape((-1, 1)))
    reshaped = scipy_backend._reshape_single_constant_tensor(a, (3, 2))
    expected = np.arange(6).reshape((3, 2), order='F')
    expected = sp.csc_matrix(np.tile(expected, (3, 1)))
    assert (reshaped != expected).nnz == 0