import itertools
import warnings
import numpy as np
from numpy import (arange, array, dot, zeros, identity, conjugate, transpose,
from numpy.random import random
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (solve, inv, det, lstsq, pinv, pinvh, norm,
from scipy.linalg._testutils import assert_no_overwrite
from scipy._lib._testutils import check_free_memory, IS_MUSL
from scipy.linalg.blas import HAS_ILP64
from scipy._lib.deprecation import _NoValue
class TestDet:

    def setup_method(self):
        self.rng = np.random.default_rng(1680305949878959)

    def test_1x1_all_singleton_dims(self):
        a = np.array([[1]])
        deta = det(a)
        assert deta.dtype.char == 'd'
        assert np.isscalar(deta)
        assert deta == 1.0
        a = np.array([[[[1]]]], dtype='f')
        deta = det(a)
        assert deta.dtype.char == 'd'
        assert np.isscalar(deta)
        assert deta == 1.0
        a = np.array([[[1 + 3j]]], dtype=np.complex64)
        deta = det(a)
        assert deta.dtype.char == 'D'
        assert np.isscalar(deta)
        assert deta == 1.0 + 3j

    def test_1by1_stacked_input_output(self):
        a = self.rng.random([4, 5, 1, 1], dtype=np.float32)
        deta = det(a)
        assert deta.dtype.char == 'd'
        assert deta.shape == (4, 5)
        assert_allclose(deta, np.squeeze(a))
        a = self.rng.random([4, 5, 1, 1], dtype=np.float32) * np.complex64(1j)
        deta = det(a)
        assert deta.dtype.char == 'D'
        assert deta.shape == (4, 5)
        assert_allclose(deta, np.squeeze(a))

    @pytest.mark.parametrize('shape', [[2, 2], [20, 20], [3, 2, 20, 20]])
    def test_simple_det_shapes_real_complex(self, shape):
        a = self.rng.uniform(-1.0, 1.0, size=shape)
        d1, d2 = (det(a), np.linalg.det(a))
        assert_allclose(d1, d2)
        b = self.rng.uniform(-1.0, 1.0, size=shape) * 1j
        b += self.rng.uniform(-0.5, 0.5, size=shape)
        d3, d4 = (det(b), np.linalg.det(b))
        assert_allclose(d3, d4)

    def test_for_known_det_values(self):
        a = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1, -1, 1, -1, 1, -1, 1, -1], [1, 1, -1, -1, 1, 1, -1, -1], [1, -1, -1, 1, 1, -1, -1, 1], [1, 1, 1, 1, -1, -1, -1, -1], [1, -1, 1, -1, -1, 1, -1, 1], [1, 1, -1, -1, -1, -1, 1, 1], [1, -1, -1, 1, -1, 1, 1, -1]])
        assert_allclose(det(a), 4096.0)
        assert_allclose(det(np.arange(25).reshape(5, 5)), 0.0)
        a = np.array([[0.0 + 0j, 0.0 + 0j, 0.0 - 1j, 1.0 - 1j], [0.0 + 0j, 0.0 + 0j, 1.0 + 0j, 0.0 - 1j], [0.0 + 1j, 1.0 + 1j, 0.0 + 0j, 0.0 + 0j], [1.0 + 0j, 0.0 + 1j, 0.0 + 0j, 0.0 + 0j]], dtype=np.complex64)
        assert_allclose(det(a), 5.0 + 0j)
        a = np.array([[-2.0, -3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -4.0, 0.0, -5.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -6.0, 0.0, -7.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -8.0, 0.0, -9.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]) * 1j
        assert_allclose(det(a), 9.0)

    @pytest.mark.parametrize('typ', [x for x in np.typecodes['All'][:20] if x not in 'gG'])
    def test_sample_compatible_dtype_input(self, typ):
        n = 4
        a = self.rng.random([n, n]).astype(typ)
        assert isinstance(det(a), (np.float64, np.complex128))

    def test_incompatible_dtype_input(self):
        msg = 'cannot be cast to float\\(32, 64\\)'
        for c, t in zip('SUO', ['bytes8', 'str32', 'object']):
            with assert_raises(TypeError, match=msg):
                det(np.array([['a', 'b']] * 2, dtype=c))
        with assert_raises(TypeError, match=msg):
            det(np.array([[b'a', b'b']] * 2, dtype='V'))
        with assert_raises(TypeError, match=msg):
            det(np.array([[100, 200]] * 2, dtype='datetime64[s]'))
        with assert_raises(TypeError, match=msg):
            det(np.array([[100, 200]] * 2, dtype='timedelta64[s]'))

    def test_empty_edge_cases(self):
        assert_allclose(det(np.empty([0, 0])), 1.0)
        assert_allclose(det(np.empty([0, 0, 0])), np.array([]))
        assert_allclose(det(np.empty([3, 0, 0])), np.array([1.0, 1.0, 1.0]))
        with assert_raises(ValueError, match='Last 2 dimensions'):
            det(np.empty([0, 0, 3]))
        with assert_raises(ValueError, match='at least two-dimensional'):
            det(np.array([]))
        with assert_raises(ValueError, match='Last 2 dimensions'):
            det(np.array([[]]))
        with assert_raises(ValueError, match='Last 2 dimensions'):
            det(np.array([[[]]]))

    def test_overwrite_a(self):
        a = np.arange(9).reshape(3, 3).astype(np.float32)
        ac = a.copy()
        deta = det(ac, overwrite_a=True)
        assert_allclose(deta, 0.0)
        assert not (a == ac).all()

    def test_readonly_array(self):
        a = np.array([[2.0, 0.0, 1.0], [5.0, 3.0, -1.0], [1.0, 1.0, 1.0]])
        a.setflags(write=False)
        assert_allclose(det(a, overwrite_a=True), 10.0)

    def test_simple_check_finite(self):
        a = [[1, 2], [3, np.inf]]
        with assert_raises(ValueError, match='array must not contain'):
            det(a)