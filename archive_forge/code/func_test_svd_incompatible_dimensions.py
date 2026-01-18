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
@pytest.mark.parametrize('ndim', [0, 1, 3])
def test_svd_incompatible_dimensions(ndim):
    with pytest.raises(ValueError, match='Array must be 2D'):
        x = da.random.default_rng().random((10,) * ndim, chunks=(-1,) * ndim)
        da.linalg.svd(x)