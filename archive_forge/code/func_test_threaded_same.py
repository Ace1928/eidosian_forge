from scipy import fft
import numpy as np
import pytest
from numpy.testing import assert_allclose
import multiprocessing
import os
@pytest.mark.parametrize('func', [fft.fft, fft.ifft, fft.fft2, fft.ifft2, fft.fftn, fft.ifftn, fft.rfft, fft.irfft, fft.rfft2, fft.irfft2, fft.rfftn, fft.irfftn, fft.hfft, fft.ihfft, fft.hfft2, fft.ihfft2, fft.hfftn, fft.ihfftn, fft.dct, fft.idct, fft.dctn, fft.idctn, fft.dst, fft.idst, fft.dstn, fft.idstn])
@pytest.mark.parametrize('workers', [2, -1])
def test_threaded_same(x, func, workers):
    expected = func(x, workers=1)
    actual = func(x, workers=workers)
    assert_allclose(actual, expected)