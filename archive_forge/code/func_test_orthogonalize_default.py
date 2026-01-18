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
@pytest.mark.parametrize('func', [dct, dst, dctn, dstn])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
def test_orthogonalize_default(func, type, xp):
    x = xp.asarray(np.random.rand(100))
    for norm, ortho in [('forward', False), ('backward', False), ('ortho', True)]:
        a = func(x, type=type, norm=norm, orthogonalize=ortho)
        b = func(x, type=type, norm=norm)
        xp_assert_close(a, b)