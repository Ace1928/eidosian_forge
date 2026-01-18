from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
@staticmethod
@pytest.mark.parametrize('shape', [(1, 1), (2, 2), (3, 2), (2, 3)])
def test_transpose_stacked(shape, scipy_backend):
    p = 2
    param_id = 2
    matrices = [sp.random(*shape, random_state=i, density=0.5) for i in range(p)]
    stacked = sp.vstack(matrices)
    transposed = scipy_backend._transpose_stacked(stacked, param_id)
    expected = sp.vstack([m.T for m in matrices])
    assert (expected != transposed).nnz == 0