from __future__ import annotations
import sys
import pytest
import numpy as np
import scipy.linalg
from packaging.version import parse as parse_version
import dask.array as da
from dask.array.linalg import qr, sfqr, svd, svd_compressed, tsqr
from dask.array.numpy_compat import _np_version
from dask.array.utils import assert_eq, same_keys, svd_flip
def test_solve_triangular_errors():
    A = np.random.default_rng().integers(0, 10, (10, 10, 10))
    b = np.random.default_rng().integers(1, 10, 10)
    dA = da.from_array(A, chunks=(5, 5, 5))
    db = da.from_array(b, chunks=5)
    pytest.raises(ValueError, lambda: da.linalg.solve_triangular(dA, db))
    A = np.random.default_rng().integers(0, 10, (10, 10))
    b = np.random.default_rng().integers(1, 10, 10)
    dA = da.from_array(A, chunks=(3, 3))
    db = da.from_array(b, chunks=5)
    pytest.raises(ValueError, lambda: da.linalg.solve_triangular(dA, db))