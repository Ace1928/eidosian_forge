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
def _check_lu_result(p, l, u, A):
    assert np.allclose(p.dot(l).dot(u), A)
    assert_eq(l, da.tril(l), check_graph=False)
    assert_eq(u, da.triu(u), check_graph=False)