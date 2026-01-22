import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
class SharedNanFunctionsTestsMixin:

    def test_mutation(self):
        ndat = _ndat.copy()
        for f in self.nanfuncs:
            f(ndat)
            assert_equal(ndat, _ndat)

    def test_keepdims(self):
        mat = np.eye(3)
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for axis in [None, 0, 1]:
                tgt = rf(mat, axis=axis, keepdims=True)
                res = nf(mat, axis=axis, keepdims=True)
                assert_(res.ndim == tgt.ndim)

    def test_out(self):
        mat = np.eye(3)
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            resout = np.zeros(3)
            tgt = rf(mat, axis=1)
            res = nf(mat, axis=1, out=resout)
            assert_almost_equal(res, resout)
            assert_almost_equal(res, tgt)

    def test_dtype_from_dtype(self):
        mat = np.eye(3)
        codes = 'efdgFDG'
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for c in codes:
                with suppress_warnings() as sup:
                    if nf in {np.nanstd, np.nanvar} and c in 'FDG':
                        sup.filter(np.ComplexWarning)
                    tgt = rf(mat, dtype=np.dtype(c), axis=1).dtype.type
                    res = nf(mat, dtype=np.dtype(c), axis=1).dtype.type
                    assert_(res is tgt)
                    tgt = rf(mat, dtype=np.dtype(c), axis=None).dtype.type
                    res = nf(mat, dtype=np.dtype(c), axis=None).dtype.type
                    assert_(res is tgt)

    def test_dtype_from_char(self):
        mat = np.eye(3)
        codes = 'efdgFDG'
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for c in codes:
                with suppress_warnings() as sup:
                    if nf in {np.nanstd, np.nanvar} and c in 'FDG':
                        sup.filter(np.ComplexWarning)
                    tgt = rf(mat, dtype=c, axis=1).dtype.type
                    res = nf(mat, dtype=c, axis=1).dtype.type
                    assert_(res is tgt)
                    tgt = rf(mat, dtype=c, axis=None).dtype.type
                    res = nf(mat, dtype=c, axis=None).dtype.type
                    assert_(res is tgt)

    def test_dtype_from_input(self):
        codes = 'efdgFDG'
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for c in codes:
                mat = np.eye(3, dtype=c)
                tgt = rf(mat, axis=1).dtype.type
                res = nf(mat, axis=1).dtype.type
                assert_(res is tgt, 'res %s, tgt %s' % (res, tgt))
                tgt = rf(mat, axis=None).dtype.type
                res = nf(mat, axis=None).dtype.type
                assert_(res is tgt)

    def test_result_values(self):
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            tgt = [rf(d) for d in _rdat]
            res = nf(_ndat, axis=1)
            assert_almost_equal(res, tgt)

    def test_scalar(self):
        for f in self.nanfuncs:
            assert_(f(0.0) == 0.0)

    def test_subclass(self):

        class MyNDArray(np.ndarray):
            pass
        array = np.eye(3)
        mine = array.view(MyNDArray)
        for f in self.nanfuncs:
            expected_shape = f(array, axis=0).shape
            res = f(mine, axis=0)
            assert_(isinstance(res, MyNDArray))
            assert_(res.shape == expected_shape)
            expected_shape = f(array, axis=1).shape
            res = f(mine, axis=1)
            assert_(isinstance(res, MyNDArray))
            assert_(res.shape == expected_shape)
            expected_shape = f(array).shape
            res = f(mine)
            assert_(isinstance(res, MyNDArray))
            assert_(res.shape == expected_shape)