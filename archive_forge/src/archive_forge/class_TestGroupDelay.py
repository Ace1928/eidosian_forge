import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestGroupDelay:

    def test_identity_filter(self):
        w, gd = group_delay((1, 1))
        assert_array_almost_equal(w, pi * np.arange(512) / 512)
        assert_array_almost_equal(gd, np.zeros(512))
        w, gd = group_delay((1, 1), whole=True)
        assert_array_almost_equal(w, 2 * pi * np.arange(512) / 512)
        assert_array_almost_equal(gd, np.zeros(512))

    def test_fir(self):
        N = 100
        b = firwin(N + 1, 0.1)
        w, gd = group_delay((b, 1))
        assert_allclose(gd, 0.5 * N)

    def test_iir(self):
        b, a = butter(4, 0.1)
        w = np.linspace(0, pi, num=10, endpoint=False)
        w, gd = group_delay((b, a), w=w)
        matlab_gd = np.array([8.249313898506037, 11.958947880907104, 2.452325615326005, 1.048918665702008, 0.611382575635897, 0.418293269460578, 0.317932917836572, 0.261371844762525, 0.229038045801298, 0.212185774208521])
        assert_array_almost_equal(gd, matlab_gd)

    def test_singular(self):
        z1 = np.exp(1j * 0.1 * pi)
        z2 = np.exp(1j * 0.25 * pi)
        p1 = np.exp(1j * 0.5 * pi)
        p2 = np.exp(1j * 0.8 * pi)
        b = np.convolve([1, -z1], [1, -z2])
        a = np.convolve([1, -p1], [1, -p2])
        w = np.array([0.1 * pi, 0.25 * pi, -0.5 * pi, -0.8 * pi])
        w, gd = assert_warns(UserWarning, group_delay, (b, a), w=w)

    def test_backward_compat(self):
        w1, gd1 = group_delay((1, 1))
        w2, gd2 = group_delay((1, 1), None)
        assert_array_almost_equal(w1, w2)
        assert_array_almost_equal(gd1, gd2)

    def test_fs_param(self):
        b, a = butter(4, 4800, fs=96000)
        w = np.linspace(0, 96000 / 2, num=10, endpoint=False)
        w, gd = group_delay((b, a), w=w, fs=96000)
        norm_gd = np.array([8.249313898506037, 11.958947880907104, 2.452325615326005, 1.048918665702008, 0.611382575635897, 0.418293269460578, 0.317932917836572, 0.261371844762525, 0.229038045801298, 0.212185774208521])
        assert_array_almost_equal(gd, norm_gd)

    def test_w_or_N_types(self):
        for N in (8, np.int8(8), np.int16(8), np.int32(8), np.int64(8), np.array(8)):
            w, gd = group_delay((1, 1), N)
            assert_array_almost_equal(w, pi * np.arange(8) / 8)
            assert_array_almost_equal(gd, np.zeros(8))
        for w in (8.0, 8.0 + 0j):
            w_out, gd = group_delay((1, 1), w)
            assert_array_almost_equal(w_out, [8])
            assert_array_almost_equal(gd, [0])