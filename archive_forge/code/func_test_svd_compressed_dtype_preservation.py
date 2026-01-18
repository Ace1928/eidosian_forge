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
@pytest.mark.parametrize('input_dtype, output_dtype', [(np.float32, np.float32), (np.float64, np.float64)])
def test_svd_compressed_dtype_preservation(input_dtype, output_dtype):
    x = da.random.default_rng().random((50, 50), chunks=(50, 50)).astype(input_dtype)
    u, s, vt = svd_compressed(x, 1, seed=4321)
    assert u.dtype == s.dtype == vt.dtype == output_dtype