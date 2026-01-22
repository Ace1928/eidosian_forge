from os.path import join, dirname
from typing import Callable, Union
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
@pytest.mark.parametrize('rdt', [np.longdouble, np.float64, np.float32, int])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
class TestDCT:

    def test_definition(self, rdt, type, fftwdata_size, reference_data):
        x, yr, dt = fftw_dct_ref(type, fftwdata_size, rdt, reference_data)
        y = dct(x, type=type)
        assert_equal(y.dtype, dt)
        dec = dec_map[dct, rdt, type]
        assert_allclose(y, yr, rtol=0.0, atol=np.max(yr) * 10 ** (-dec))

    @pytest.mark.parametrize('size', [7, 8, 9, 16, 32, 64])
    def test_axis(self, rdt, type, size):
        nt = 2
        dec = dec_map[dct, rdt, type]
        x = np.random.randn(nt, size)
        y = dct(x, type=type)
        for j in range(nt):
            assert_array_almost_equal(y[j], dct(x[j], type=type), decimal=dec)
        x = x.T
        y = dct(x, axis=0, type=type)
        for j in range(nt):
            assert_array_almost_equal(y[:, j], dct(x[:, j], type=type), decimal=dec)