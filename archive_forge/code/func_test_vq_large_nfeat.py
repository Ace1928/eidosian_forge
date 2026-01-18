import warnings
import sys
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.cluster.vq import (kmeans, kmeans2, py_vq, vq, whiten,
from scipy.cluster import _vq
from scipy.conftest import (
from scipy.sparse._sputils import matrix
from scipy._lib._array_api import (
@skip_if_array_api_gpu
@array_api_compatible
def test_vq_large_nfeat(self, xp):
    X = np.random.rand(20, 20)
    code_book = np.random.rand(3, 20)
    codes0, dis0 = _vq.vq(X, code_book)
    codes1, dis1 = py_vq(xp.asarray(X), xp.asarray(code_book))
    xp_assert_close(dis1, xp.asarray(dis0), rtol=1e-05)
    xp_assert_equal(codes1, xp.asarray(codes0, dtype=xp.int64), check_dtype=False)
    X = X.astype(np.float32)
    code_book = code_book.astype(np.float32)
    codes0, dis0 = _vq.vq(X, code_book)
    codes1, dis1 = py_vq(xp.asarray(X), xp.asarray(code_book))
    xp_assert_close(dis1, xp.asarray(dis0, dtype=xp.float64), rtol=1e-05)
    xp_assert_equal(codes1, xp.asarray(codes0, dtype=xp.int64), check_dtype=False)