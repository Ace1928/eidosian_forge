import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from itertools import product
from math import gcd
import pytest
from pytest import raises as assert_raises
from numpy.testing import (
from numpy import array, arange
import numpy as np
from scipy.fft import fft
from scipy.ndimage import correlate1d
from scipy.optimize import fmin, linear_sum_assignment
from scipy import signal
from scipy.signal import (
from scipy.signal.windows import hann
from scipy.signal._signaltools import (_filtfilt_gust, _compute_factors,
from scipy.signal._upfirdn import _upfirdn_modes
from scipy._lib import _testutils
from scipy._lib._util import ComplexWarning, np_long, np_ulong
class _TestLinearFilter:

    def generate(self, shape):
        x = np.linspace(0, np.prod(shape) - 1, np.prod(shape)).reshape(shape)
        return self.convert_dtype(x)

    def convert_dtype(self, arr):
        if self.dtype == np.dtype('O'):
            arr = np.asarray(arr)
            out = np.empty(arr.shape, self.dtype)
            iter = np.nditer([arr, out], ['refs_ok', 'zerosize_ok'], [['readonly'], ['writeonly']])
            for x, y in iter:
                y[...] = self.type(x[()])
            return out
        else:
            return np.array(arr, self.dtype, copy=False)

    def test_rank_1_IIR(self):
        x = self.generate((6,))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, -0.5])
        y_r = self.convert_dtype([0, 2, 4, 6, 8, 10.0])
        assert_array_almost_equal(lfilter(b, a, x), y_r)

    def test_rank_1_FIR(self):
        x = self.generate((6,))
        b = self.convert_dtype([1, 1])
        a = self.convert_dtype([1])
        y_r = self.convert_dtype([0, 1, 3, 5, 7, 9.0])
        assert_array_almost_equal(lfilter(b, a, x), y_r)

    def test_rank_1_IIR_init_cond(self):
        x = self.generate((6,))
        b = self.convert_dtype([1, 0, -1])
        a = self.convert_dtype([0.5, -0.5])
        zi = self.convert_dtype([1, 2])
        y_r = self.convert_dtype([1, 5, 9, 13, 17, 21])
        zf_r = self.convert_dtype([13, -10])
        y, zf = lfilter(b, a, x, zi=zi)
        assert_array_almost_equal(y, y_r)
        assert_array_almost_equal(zf, zf_r)

    def test_rank_1_FIR_init_cond(self):
        x = self.generate((6,))
        b = self.convert_dtype([1, 1, 1])
        a = self.convert_dtype([1])
        zi = self.convert_dtype([1, 1])
        y_r = self.convert_dtype([1, 2, 3, 6, 9, 12.0])
        zf_r = self.convert_dtype([9, 5])
        y, zf = lfilter(b, a, x, zi=zi)
        assert_array_almost_equal(y, y_r)
        assert_array_almost_equal(zf, zf_r)

    def test_rank_2_IIR_axis_0(self):
        x = self.generate((4, 3))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, 0.5])
        y_r2_a0 = self.convert_dtype([[0, 2, 4], [6, 4, 2], [0, 2, 4], [6, 4, 2]])
        y = lfilter(b, a, x, axis=0)
        assert_array_almost_equal(y_r2_a0, y)

    def test_rank_2_IIR_axis_1(self):
        x = self.generate((4, 3))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, 0.5])
        y_r2_a1 = self.convert_dtype([[0, 2, 0], [6, -4, 6], [12, -10, 12], [18, -16, 18]])
        y = lfilter(b, a, x, axis=1)
        assert_array_almost_equal(y_r2_a1, y)

    def test_rank_2_IIR_axis_0_init_cond(self):
        x = self.generate((4, 3))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, 0.5])
        zi = self.convert_dtype(np.ones((4, 1)))
        y_r2_a0_1 = self.convert_dtype([[1, 1, 1], [7, -5, 7], [13, -11, 13], [19, -17, 19]])
        zf_r = self.convert_dtype([-5, -17, -29, -41])[:, np.newaxis]
        y, zf = lfilter(b, a, x, axis=1, zi=zi)
        assert_array_almost_equal(y_r2_a0_1, y)
        assert_array_almost_equal(zf, zf_r)

    def test_rank_2_IIR_axis_1_init_cond(self):
        x = self.generate((4, 3))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, 0.5])
        zi = self.convert_dtype(np.ones((1, 3)))
        y_r2_a0_0 = self.convert_dtype([[1, 3, 5], [5, 3, 1], [1, 3, 5], [5, 3, 1]])
        zf_r = self.convert_dtype([[-23, -23, -23]])
        y, zf = lfilter(b, a, x, axis=0, zi=zi)
        assert_array_almost_equal(y_r2_a0_0, y)
        assert_array_almost_equal(zf, zf_r)

    def test_rank_3_IIR(self):
        x = self.generate((4, 3, 2))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, 0.5])
        for axis in range(x.ndim):
            y = lfilter(b, a, x, axis)
            y_r = np.apply_along_axis(lambda w: lfilter(b, a, w), axis, x)
            assert_array_almost_equal(y, y_r)

    def test_rank_3_IIR_init_cond(self):
        x = self.generate((4, 3, 2))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, 0.5])
        for axis in range(x.ndim):
            zi_shape = list(x.shape)
            zi_shape[axis] = 1
            zi = self.convert_dtype(np.ones(zi_shape))
            zi1 = self.convert_dtype([1])
            y, zf = lfilter(b, a, x, axis, zi)

            def lf0(w):
                return lfilter(b, a, w, zi=zi1)[0]

            def lf1(w):
                return lfilter(b, a, w, zi=zi1)[1]
            y_r = np.apply_along_axis(lf0, axis, x)
            zf_r = np.apply_along_axis(lf1, axis, x)
            assert_array_almost_equal(y, y_r)
            assert_array_almost_equal(zf, zf_r)

    def test_rank_3_FIR(self):
        x = self.generate((4, 3, 2))
        b = self.convert_dtype([1, 0, -1])
        a = self.convert_dtype([1])
        for axis in range(x.ndim):
            y = lfilter(b, a, x, axis)
            y_r = np.apply_along_axis(lambda w: lfilter(b, a, w), axis, x)
            assert_array_almost_equal(y, y_r)

    def test_rank_3_FIR_init_cond(self):
        x = self.generate((4, 3, 2))
        b = self.convert_dtype([1, 0, -1])
        a = self.convert_dtype([1])
        for axis in range(x.ndim):
            zi_shape = list(x.shape)
            zi_shape[axis] = 2
            zi = self.convert_dtype(np.ones(zi_shape))
            zi1 = self.convert_dtype([1, 1])
            y, zf = lfilter(b, a, x, axis, zi)

            def lf0(w):
                return lfilter(b, a, w, zi=zi1)[0]

            def lf1(w):
                return lfilter(b, a, w, zi=zi1)[1]
            y_r = np.apply_along_axis(lf0, axis, x)
            zf_r = np.apply_along_axis(lf1, axis, x)
            assert_array_almost_equal(y, y_r)
            assert_array_almost_equal(zf, zf_r)

    def test_zi_pseudobroadcast(self):
        x = self.generate((4, 5, 20))
        b, a = signal.butter(8, 0.2, output='ba')
        b = self.convert_dtype(b)
        a = self.convert_dtype(a)
        zi_size = b.shape[0] - 1
        zi_full = self.convert_dtype(np.ones((4, 5, zi_size)))
        zi_sing = self.convert_dtype(np.ones((1, 1, zi_size)))
        y_full, zf_full = lfilter(b, a, x, zi=zi_full)
        y_sing, zf_sing = lfilter(b, a, x, zi=zi_sing)
        assert_array_almost_equal(y_sing, y_full)
        assert_array_almost_equal(zf_full, zf_sing)
        assert_raises(ValueError, lfilter, b, a, x, -1, np.ones(zi_size))

    def test_scalar_a(self):
        x = self.generate(6)
        b = self.convert_dtype([1, 0, -1])
        a = self.convert_dtype([1])
        y_r = self.convert_dtype([0, 1, 2, 2, 2, 2])
        y = lfilter(b, a[0], x)
        assert_array_almost_equal(y, y_r)

    def test_zi_some_singleton_dims(self):
        x = self.convert_dtype(np.zeros((3, 2, 5), 'l'))
        b = self.convert_dtype(np.ones(5, 'l'))
        a = self.convert_dtype(np.array([1, 0, 0]))
        zi = np.ones((3, 1, 4), 'l')
        zi[1, :, :] *= 2
        zi[2, :, :] *= 3
        zi = self.convert_dtype(zi)
        zf_expected = self.convert_dtype(np.zeros((3, 2, 4), 'l'))
        y_expected = np.zeros((3, 2, 5), 'l')
        y_expected[:, :, :4] = [[[1]], [[2]], [[3]]]
        y_expected = self.convert_dtype(y_expected)
        y_iir, zf_iir = lfilter(b, a, x, -1, zi)
        assert_array_almost_equal(y_iir, y_expected)
        assert_array_almost_equal(zf_iir, zf_expected)
        y_fir, zf_fir = lfilter(b, a[0], x, -1, zi)
        assert_array_almost_equal(y_fir, y_expected)
        assert_array_almost_equal(zf_fir, zf_expected)

    def base_bad_size_zi(self, b, a, x, axis, zi):
        b = self.convert_dtype(b)
        a = self.convert_dtype(a)
        x = self.convert_dtype(x)
        zi = self.convert_dtype(zi)
        assert_raises(ValueError, lfilter, b, a, x, axis, zi)

    def test_bad_size_zi(self):
        x1 = np.arange(6)
        self.base_bad_size_zi([1], [1], x1, -1, [1])
        self.base_bad_size_zi([1, 1], [1], x1, -1, [0, 1])
        self.base_bad_size_zi([1, 1], [1], x1, -1, [[0]])
        self.base_bad_size_zi([1, 1], [1], x1, -1, [0, 1, 2])
        self.base_bad_size_zi([1, 1, 1], [1], x1, -1, [[0]])
        self.base_bad_size_zi([1, 1, 1], [1], x1, -1, [0, 1, 2])
        self.base_bad_size_zi([1], [1, 1], x1, -1, [0, 1])
        self.base_bad_size_zi([1], [1, 1], x1, -1, [[0]])
        self.base_bad_size_zi([1], [1, 1], x1, -1, [0, 1, 2])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x1, -1, [0])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x1, -1, [[0], [1]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x1, -1, [0, 1, 2])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x1, -1, [0, 1, 2, 3])
        self.base_bad_size_zi([1, 1], [1, 1, 1], x1, -1, [0])
        self.base_bad_size_zi([1, 1], [1, 1, 1], x1, -1, [[0], [1]])
        self.base_bad_size_zi([1, 1], [1, 1, 1], x1, -1, [0, 1, 2])
        self.base_bad_size_zi([1, 1], [1, 1, 1], x1, -1, [0, 1, 2, 3])
        x2 = np.arange(12).reshape((4, 3))
        self.base_bad_size_zi([1], [1], x2, 0, [0])
        self.base_bad_size_zi([1, 1], [1], x2, 0, [0, 1, 2])
        self.base_bad_size_zi([1, 1], [1], x2, 0, [[[0, 1, 2]]])
        self.base_bad_size_zi([1, 1], [1], x2, 0, [[0], [1], [2]])
        self.base_bad_size_zi([1, 1], [1], x2, 0, [[0, 1]])
        self.base_bad_size_zi([1, 1], [1], x2, 0, [[0, 1, 2, 3]])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 0, [0, 1, 2, 3, 4, 5])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 0, [[[0, 1, 2], [3, 4, 5]]])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 0, [[0, 1], [2, 3], [4, 5]])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 0, [[0, 1], [2, 3]])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 0, [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.base_bad_size_zi([1], [1, 1], x2, 0, [0, 1, 2])
        self.base_bad_size_zi([1], [1, 1], x2, 0, [[[0, 1, 2]]])
        self.base_bad_size_zi([1], [1, 1], x2, 0, [[0], [1], [2]])
        self.base_bad_size_zi([1], [1, 1], x2, 0, [[0, 1]])
        self.base_bad_size_zi([1], [1, 1], x2, 0, [[0, 1, 2, 3]])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 0, [0, 1, 2, 3, 4, 5])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 0, [[[0, 1, 2], [3, 4, 5]]])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 0, [[0, 1], [2, 3], [4, 5]])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 0, [[0, 1], [2, 3]])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 0, [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 0, [0, 1, 2, 3, 4, 5])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 0, [[[0, 1, 2], [3, 4, 5]]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 0, [[0, 1], [2, 3], [4, 5]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 0, [[0, 1], [2, 3]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 0, [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.base_bad_size_zi([1], [1], x2, 1, [0])
        self.base_bad_size_zi([1, 1], [1], x2, 1, [0, 1, 2, 3])
        self.base_bad_size_zi([1, 1], [1], x2, 1, [[[0], [1], [2], [3]]])
        self.base_bad_size_zi([1, 1], [1], x2, 1, [[0, 1, 2, 3]])
        self.base_bad_size_zi([1, 1], [1], x2, 1, [[0], [1], [2]])
        self.base_bad_size_zi([1, 1], [1], x2, 1, [[0], [1], [2], [3], [4]])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 1, [0, 1, 2, 3, 4, 5, 6, 7])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 1, [[[0, 1], [2, 3], [4, 5], [6, 7]]])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 1, [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 1, [[0, 1], [2, 3], [4, 5]])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 1, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        self.base_bad_size_zi([1], [1, 1], x2, 1, [0, 1, 2, 3])
        self.base_bad_size_zi([1], [1, 1], x2, 1, [[[0], [1], [2], [3]]])
        self.base_bad_size_zi([1], [1, 1], x2, 1, [[0, 1, 2, 3]])
        self.base_bad_size_zi([1], [1, 1], x2, 1, [[0], [1], [2]])
        self.base_bad_size_zi([1], [1, 1], x2, 1, [[0], [1], [2], [3], [4]])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 1, [0, 1, 2, 3, 4, 5, 6, 7])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 1, [[[0, 1], [2, 3], [4, 5], [6, 7]]])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 1, [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 1, [[0, 1], [2, 3], [4, 5]])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 1, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 1, [0, 1, 2, 3, 4, 5, 6, 7])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 1, [[[0, 1], [2, 3], [4, 5], [6, 7]]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 1, [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 1, [[0, 1], [2, 3], [4, 5]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 1, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])

    def test_empty_zi(self):
        x = self.generate((5,))
        a = self.convert_dtype([1])
        b = self.convert_dtype([1])
        zi = self.convert_dtype([])
        y, zf = lfilter(b, a, x, zi=zi)
        assert_array_almost_equal(y, x)
        assert_equal(zf.dtype, self.dtype)
        assert_equal(zf.size, 0)

    def test_lfiltic_bad_zi(self):
        a = self.convert_dtype([1])
        b = self.convert_dtype([1])
        zi = lfiltic(b, a, [1.0, 0])
        zi_1 = lfiltic(b, a, [1, 0])
        zi_2 = lfiltic(b, a, [True, False])
        assert_array_equal(zi, zi_1)
        assert_array_equal(zi, zi_2)

    def test_short_x_FIR(self):
        a = self.convert_dtype([1])
        b = self.convert_dtype([1, 0, -1])
        zi = self.convert_dtype([2, 7])
        x = self.convert_dtype([72])
        ye = self.convert_dtype([74])
        zfe = self.convert_dtype([7, -72])
        y, zf = lfilter(b, a, x, zi=zi)
        assert_array_almost_equal(y, ye)
        assert_array_almost_equal(zf, zfe)

    def test_short_x_IIR(self):
        a = self.convert_dtype([1, 1])
        b = self.convert_dtype([1, 0, -1])
        zi = self.convert_dtype([2, 7])
        x = self.convert_dtype([72])
        ye = self.convert_dtype([74])
        zfe = self.convert_dtype([-67, -72])
        y, zf = lfilter(b, a, x, zi=zi)
        assert_array_almost_equal(y, ye)
        assert_array_almost_equal(zf, zfe)

    def test_do_not_modify_a_b_IIR(self):
        x = self.generate((6,))
        b = self.convert_dtype([1, -1])
        b0 = b.copy()
        a = self.convert_dtype([0.5, -0.5])
        a0 = a.copy()
        y_r = self.convert_dtype([0, 2, 4, 6, 8, 10.0])
        y_f = lfilter(b, a, x)
        assert_array_almost_equal(y_f, y_r)
        assert_equal(b, b0)
        assert_equal(a, a0)

    def test_do_not_modify_a_b_FIR(self):
        x = self.generate((6,))
        b = self.convert_dtype([1, 0, 1])
        b0 = b.copy()
        a = self.convert_dtype([2])
        a0 = a.copy()
        y_r = self.convert_dtype([0, 0.5, 1, 2, 3, 4.0])
        y_f = lfilter(b, a, x)
        assert_array_almost_equal(y_f, y_r)
        assert_equal(b, b0)
        assert_equal(a, a0)