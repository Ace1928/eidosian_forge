import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.fft import dct, idct, dctn, idctn, dst, idst, dstn, idstn
import scipy.fft as fft
from scipy import fftpack
from scipy.conftest import (
from scipy._lib._array_api import copy, xp_assert_close
import math
@skip_if_array_api_gpu
@array_api_compatible
@pytest.mark.parametrize('norm', ['backward', 'ortho', 'forward'])
@pytest.mark.parametrize('func', [dct, dst])
def test_orthogonalize_dcst3(func, norm, xp):
    x = xp.asarray(np.random.rand(100))
    x2 = copy(x, xp=xp)
    x2[0 if func == dct else -1] *= SQRT_2
    y1 = func(x, type=3, norm=norm, orthogonalize=True)
    y2 = func(x2, type=3, norm=norm, orthogonalize=False)
    xp_assert_close(y1, y2)