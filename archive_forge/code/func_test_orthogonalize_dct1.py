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
def test_orthogonalize_dct1(norm, xp):
    x = xp.asarray(np.random.rand(100))
    x2 = copy(x, xp=xp)
    x2[0] *= SQRT_2
    x2[-1] *= SQRT_2
    y1 = dct(x, type=1, norm=norm, orthogonalize=True)
    y2 = dct(x2, type=1, norm=norm, orthogonalize=False)
    y2[0] /= SQRT_2
    y2[-1] /= SQRT_2
    xp_assert_close(y1, y2)