from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
@staticmethod
@pytest.mark.parametrize('shape', [(1, 1), (2, 2), (3, 3), (4, 4)])
def test_stacked_kron_l(shape, scipy_backend):
    p = 2
    reps = 3
    param_id = 2
    matrices = [sp.random(*shape, random_state=i, density=0.5) for i in range(p)]
    stacked = sp.vstack(matrices)
    repeated = scipy_backend._stacked_kron_l({param_id: stacked}, reps)
    repeated = repeated[param_id]
    expected = sp.vstack([sp.kron(m, sp.eye(reps)) for m in matrices])
    assert (expected != repeated).nnz == 0