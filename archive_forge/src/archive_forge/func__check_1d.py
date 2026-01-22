from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
def _check_1d(self, routine, dtype, shape, axis, overwritable_dtypes, fftsize, overwrite_x):
    np.random.seed(1234)
    if np.issubdtype(dtype, np.complexfloating):
        data = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    else:
        data = np.random.randn(*shape)
    data = data.astype(dtype)
    should_overwrite = overwrite_x and dtype in overwritable_dtypes and (fftsize <= shape[axis])
    self._check(data, routine, fftsize, axis, overwrite_x=overwrite_x, should_overwrite=should_overwrite)