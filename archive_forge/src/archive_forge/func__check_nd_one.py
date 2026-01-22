from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
def _check_nd_one(self, routine, dtype, shape, axes, overwritable_dtypes, overwrite_x):
    np.random.seed(1234)
    if np.issubdtype(dtype, np.complexfloating):
        data = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    else:
        data = np.random.randn(*shape)
    data = data.astype(dtype)

    def fftshape_iter(shp):
        if len(shp) <= 0:
            yield ()
        else:
            for j in (shp[0] // 2, shp[0], shp[0] * 2):
                for rest in fftshape_iter(shp[1:]):
                    yield ((j,) + rest)

    def part_shape(shape, axes):
        if axes is None:
            return shape
        else:
            return tuple(np.take(shape, axes))

    def should_overwrite(data, shape, axes):
        s = part_shape(data.shape, axes)
        return overwrite_x and np.prod(shape) <= np.prod(s) and (dtype in overwritable_dtypes)
    for fftshape in fftshape_iter(part_shape(shape, axes)):
        self._check(data, routine, fftshape, axes, overwrite_x=overwrite_x, should_overwrite=should_overwrite(data, fftshape, axes))
        if data.ndim > 1:
            self._check(data.T, routine, fftshape, axes, overwrite_x=overwrite_x, should_overwrite=should_overwrite(data.T, fftshape, axes))