import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
class TestNanFunctions_CumSumProd(SharedNanFunctionsTestsMixin):
    nanfuncs = [np.nancumsum, np.nancumprod]
    stdfuncs = [np.cumsum, np.cumprod]

    @pytest.mark.parametrize('axis', [None, 0, 1])
    @pytest.mark.parametrize('dtype', np.typecodes['AllFloat'])
    @pytest.mark.parametrize('array', [np.array(np.nan), np.full((3, 3), np.nan)], ids=['0d', '2d'])
    def test_allnans(self, axis, dtype, array):
        if axis is not None and array.ndim == 0:
            pytest.skip(f'`axis != None` not supported for 0d arrays')
        array = array.astype(dtype)
        for func, identity in zip(self.nanfuncs, [0, 1]):
            out = func(array)
            assert np.all(out == identity)
            assert out.dtype == array.dtype

    def test_empty(self):
        for f, tgt_value in zip(self.nanfuncs, [0, 1]):
            mat = np.zeros((0, 3))
            tgt = tgt_value * np.ones((0, 3))
            res = f(mat, axis=0)
            assert_equal(res, tgt)
            tgt = mat
            res = f(mat, axis=1)
            assert_equal(res, tgt)
            tgt = np.zeros(0)
            res = f(mat, axis=None)
            assert_equal(res, tgt)

    def test_keepdims(self):
        for f, g in zip(self.nanfuncs, self.stdfuncs):
            mat = np.eye(3)
            for axis in [None, 0, 1]:
                tgt = f(mat, axis=axis, out=None)
                res = g(mat, axis=axis, out=None)
                assert_(res.ndim == tgt.ndim)
        for f in self.nanfuncs:
            d = np.ones((3, 5, 7, 11))
            rs = np.random.RandomState(0)
            d[rs.rand(*d.shape) < 0.5] = np.nan
            res = f(d, axis=None)
            assert_equal(res.shape, (1155,))
            for axis in np.arange(4):
                res = f(d, axis=axis)
                assert_equal(res.shape, (3, 5, 7, 11))

    def test_result_values(self):
        for axis in (-2, -1, 0, 1, None):
            tgt = np.cumprod(_ndat_ones, axis=axis)
            res = np.nancumprod(_ndat, axis=axis)
            assert_almost_equal(res, tgt)
            tgt = np.cumsum(_ndat_zeros, axis=axis)
            res = np.nancumsum(_ndat, axis=axis)
            assert_almost_equal(res, tgt)

    def test_out(self):
        mat = np.eye(3)
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            resout = np.eye(3)
            for axis in (-2, -1, 0, 1):
                tgt = rf(mat, axis=axis)
                res = nf(mat, axis=axis, out=resout)
                assert_almost_equal(res, resout)
                assert_almost_equal(res, tgt)