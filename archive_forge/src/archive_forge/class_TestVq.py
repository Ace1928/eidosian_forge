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
class TestVq:

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_py_vq(self, xp):
        initc = np.concatenate([[X[0]], [X[1]], [X[2]]])
        arrays = [xp.asarray] if SCIPY_ARRAY_API else [np.asarray, matrix]
        for tp in arrays:
            label1 = py_vq(tp(X), tp(initc))[0]
            xp_assert_equal(label1, xp.asarray(LABEL1, dtype=xp.int64), check_dtype=False)

    @skip_if_array_api
    def test_vq(self):
        initc = np.concatenate([[X[0]], [X[1]], [X[2]]])
        for tp in [np.asarray, matrix]:
            label1, dist = _vq.vq(tp(X), tp(initc))
            assert_array_equal(label1, LABEL1)
            tlabel1, tdist = vq(tp(X), tp(initc))

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_vq_1d(self, xp):
        data = X[:, 0]
        initc = data[:3]
        a, b = _vq.vq(data, initc)
        data = xp.asarray(data)
        initc = xp.asarray(initc)
        ta, tb = py_vq(data[:, np.newaxis], initc[:, np.newaxis])
        xp_assert_equal(ta, xp.asarray(a, dtype=xp.int64), check_dtype=False)
        xp_assert_equal(tb, xp.asarray(b))

    @skip_if_array_api
    def test__vq_sametype(self):
        a = np.array([1.0, 2.0], dtype=np.float64)
        b = a.astype(np.float32)
        assert_raises(TypeError, _vq.vq, a, b)

    @skip_if_array_api
    def test__vq_invalid_type(self):
        a = np.array([1, 2], dtype=int)
        assert_raises(TypeError, _vq.vq, a, a)

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

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_vq_large_features(self, xp):
        X = np.random.rand(10, 5) * 1000000
        code_book = np.random.rand(2, 5) * 1000000
        codes0, dis0 = _vq.vq(X, code_book)
        codes1, dis1 = py_vq(xp.asarray(X), xp.asarray(code_book))
        xp_assert_close(dis1, xp.asarray(dis0), rtol=1e-05)
        xp_assert_equal(codes1, xp.asarray(codes0, dtype=xp.int64), check_dtype=False)