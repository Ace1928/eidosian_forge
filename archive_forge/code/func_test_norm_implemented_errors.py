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
@pytest.mark.parametrize('shape, chunks, axis', [[(3, 2, 4), (2, 2, 2), (1, 2)], [(2, 3, 4, 5), (2, 2, 2, 2), (-1, -2)]])
@pytest.mark.parametrize('norm', ['nuc', 2, -2])
@pytest.mark.parametrize('keepdims', [False, True])
def test_norm_implemented_errors(shape, chunks, axis, norm, keepdims):
    a = np.random.default_rng().random(shape)
    d = da.from_array(a, chunks=chunks)
    if len(shape) > 2 and len(axis) == 2:
        with pytest.raises(NotImplementedError):
            da.linalg.norm(d, ord=norm, axis=axis, keepdims=keepdims)