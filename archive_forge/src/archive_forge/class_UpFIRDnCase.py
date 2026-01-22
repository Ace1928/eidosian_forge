import numpy as np
from itertools import product
from numpy.testing import assert_equal, assert_allclose
from pytest import raises as assert_raises
import pytest
from scipy.signal import upfirdn, firwin
from scipy.signal._upfirdn import _output_len, _upfirdn_modes
from scipy.signal._upfirdn_apply import _pad_test
class UpFIRDnCase:
    """Test _UpFIRDn object"""

    def __init__(self, up, down, h, x_dtype):
        self.up = up
        self.down = down
        self.h = np.atleast_1d(h)
        self.x_dtype = x_dtype
        self.rng = np.random.RandomState(17)

    def __call__(self):
        self.scrub(np.ones(1, self.x_dtype))
        self.scrub(np.ones(10, self.x_dtype))
        x = self.rng.randn(10).astype(self.x_dtype)
        if self.x_dtype in (np.complex64, np.complex128):
            x += 1j * self.rng.randn(10)
        self.scrub(x)
        self.scrub(np.arange(10).astype(self.x_dtype))
        size = (2, 3, 5)
        x = self.rng.randn(*size).astype(self.x_dtype)
        if self.x_dtype in (np.complex64, np.complex128):
            x += 1j * self.rng.randn(*size)
        for axis in range(len(size)):
            self.scrub(x, axis=axis)
        x = x[:, ::2, 1::3].T
        for axis in range(len(size)):
            self.scrub(x, axis=axis)

    def scrub(self, x, axis=-1):
        yr = np.apply_along_axis(upfirdn_naive, axis, x, self.h, self.up, self.down)
        want_len = _output_len(len(self.h), x.shape[axis], self.up, self.down)
        assert yr.shape[axis] == want_len
        y = upfirdn(self.h, x, self.up, self.down, axis=axis)
        assert y.shape[axis] == want_len
        assert y.shape == yr.shape
        dtypes = (self.h.dtype, x.dtype)
        if all((d == np.complex64 for d in dtypes)):
            assert_equal(y.dtype, np.complex64)
        elif np.complex64 in dtypes and np.float32 in dtypes:
            assert_equal(y.dtype, np.complex64)
        elif all((d == np.float32 for d in dtypes)):
            assert_equal(y.dtype, np.float32)
        elif np.complex128 in dtypes or np.complex64 in dtypes:
            assert_equal(y.dtype, np.complex128)
        else:
            assert_equal(y.dtype, np.float64)
        assert_allclose(yr, y)