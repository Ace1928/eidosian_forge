import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
class TestNanFunctions_SumProd(SharedNanFunctionsTestsMixin):
    nanfuncs = [np.nansum, np.nanprod]
    stdfuncs = [np.sum, np.prod]

    @pytest.mark.parametrize('axis', [None, 0, 1])
    @pytest.mark.parametrize('dtype', np.typecodes['AllFloat'])
    @pytest.mark.parametrize('array', [np.array(np.nan), np.full((3, 3), np.nan)], ids=['0d', '2d'])
    def test_allnans(self, axis, dtype, array):
        if axis is not None and array.ndim == 0:
            pytest.skip(f'`axis != None` not supported for 0d arrays')
        array = array.astype(dtype)
        for func, identity in zip(self.nanfuncs, [0, 1]):
            out = func(array, axis=axis)
            assert np.all(out == identity)
            assert out.dtype == array.dtype

    def test_empty(self):
        for f, tgt_value in zip([np.nansum, np.nanprod], [0, 1]):
            mat = np.zeros((0, 3))
            tgt = [tgt_value] * 3
            res = f(mat, axis=0)
            assert_equal(res, tgt)
            tgt = []
            res = f(mat, axis=1)
            assert_equal(res, tgt)
            tgt = tgt_value
            res = f(mat, axis=None)
            assert_equal(res, tgt)

    @pytest.mark.parametrize('dtype', np.typecodes['AllFloat'])
    def test_initial(self, dtype):
        ar = np.arange(9).astype(dtype)
        ar[:5] = np.nan
        for f in self.nanfuncs:
            reference = 28 if f is np.nansum else 3360
            ret = f(ar, initial=2)
            assert ret.dtype == dtype
            assert ret == reference

    @pytest.mark.parametrize('dtype', np.typecodes['AllFloat'])
    def test_where(self, dtype):
        ar = np.arange(9).reshape(3, 3).astype(dtype)
        ar[0, :] = np.nan
        where = np.ones_like(ar, dtype=np.bool_)
        where[:, 0] = False
        for f in self.nanfuncs:
            reference = 26 if f is np.nansum else 2240
            ret = f(ar, where=where, initial=2)
            assert ret.dtype == dtype
            assert ret == reference