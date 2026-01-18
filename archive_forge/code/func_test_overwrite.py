from os.path import join, dirname
from typing import Callable, Union
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
@pytest.mark.parametrize('routine', [dct, dst, idct, idst])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
@pytest.mark.parametrize('shape, axis', [((16,), -1), ((16, 2), 0), ((2, 16), 1)])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
@pytest.mark.parametrize('overwrite_x', [True, False])
@pytest.mark.parametrize('norm', [None, 'ortho'])
def test_overwrite(routine, dtype, shape, axis, type, norm, overwrite_x):
    np.random.seed(1234)
    if np.issubdtype(dtype, np.complexfloating):
        x = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    else:
        x = np.random.randn(*shape)
    x = x.astype(dtype)
    x2 = x.copy()
    routine(x2, type, None, axis, norm, overwrite_x=overwrite_x)
    sig = '{}({}{!r}, {!r}, axis={!r}, overwrite_x={!r})'.format(routine.__name__, x.dtype, x.shape, None, axis, overwrite_x)
    if not overwrite_x:
        assert_equal(x2, x, err_msg='spurious overwrite in %s' % sig)