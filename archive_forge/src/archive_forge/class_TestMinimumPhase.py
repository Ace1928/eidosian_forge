import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
class TestMinimumPhase:

    def test_bad_args(self):
        assert_raises(ValueError, minimum_phase, [1.0])
        assert_raises(ValueError, minimum_phase, [1.0, 1.0])
        assert_raises(ValueError, minimum_phase, np.full(10, 1j))
        assert_raises(ValueError, minimum_phase, 'foo')
        assert_raises(ValueError, minimum_phase, np.ones(10), n_fft=8)
        assert_raises(ValueError, minimum_phase, np.ones(10), method='foo')
        assert_warns(RuntimeWarning, minimum_phase, np.arange(3))

    def test_homomorphic(self):
        h = [1, -1]
        h_new = minimum_phase(np.convolve(h, h[::-1]))
        assert_allclose(h_new, h, rtol=0.05)
        rng = np.random.RandomState(0)
        for n in (2, 3, 10, 11, 15, 16, 17, 20, 21, 100, 101):
            h = rng.randn(n)
            h_new = minimum_phase(np.convolve(h, h[::-1]))
            assert_allclose(np.abs(fft(h_new)), np.abs(fft(h)), rtol=0.0001)

    def test_hilbert(self):
        h = remez(12, [0, 0.3, 0.5, 1], [1, 0], fs=2.0)
        k = [0.349585548646686, 0.373552164395447, 0.326082685363438, 0.077152207480935, -0.129943946349364, -0.059355880509749]
        m = minimum_phase(h, 'hilbert')
        assert_allclose(m, k, rtol=0.005)
        h = remez(21, [0, 0.8, 0.9, 1], [0, 1], fs=2.0)
        k = [0.232486803906329, -0.133551833687071, 0.151871456867244, -0.157957283165866, 0.151739294892963, -0.12929314670509, 0.100787844523204, -0.065832656741252, 0.035361328741024, -0.014977068692269, -0.158416139047557]
        m = minimum_phase(h, 'hilbert', n_fft=2 ** 19)
        assert_allclose(m, k, rtol=0.002)