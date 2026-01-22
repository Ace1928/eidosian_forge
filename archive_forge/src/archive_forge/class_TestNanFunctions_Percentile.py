import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
class TestNanFunctions_Percentile:

    def test_mutation(self):
        ndat = _ndat.copy()
        np.nanpercentile(ndat, 30)
        assert_equal(ndat, _ndat)

    def test_keepdims(self):
        mat = np.eye(3)
        for axis in [None, 0, 1]:
            tgt = np.percentile(mat, 70, axis=axis, out=None, overwrite_input=False)
            res = np.nanpercentile(mat, 70, axis=axis, out=None, overwrite_input=False)
            assert_(res.ndim == tgt.ndim)
        d = np.ones((3, 5, 7, 11))
        w = np.random.random((4, 200)) * np.array(d.shape)[:, None]
        w = w.astype(np.intp)
        d[tuple(w)] = np.nan
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            res = np.nanpercentile(d, 90, axis=None, keepdims=True)
            assert_equal(res.shape, (1, 1, 1, 1))
            res = np.nanpercentile(d, 90, axis=(0, 1), keepdims=True)
            assert_equal(res.shape, (1, 1, 7, 11))
            res = np.nanpercentile(d, 90, axis=(0, 3), keepdims=True)
            assert_equal(res.shape, (1, 5, 7, 1))
            res = np.nanpercentile(d, 90, axis=(1,), keepdims=True)
            assert_equal(res.shape, (3, 1, 7, 11))
            res = np.nanpercentile(d, 90, axis=(0, 1, 2, 3), keepdims=True)
            assert_equal(res.shape, (1, 1, 1, 1))
            res = np.nanpercentile(d, 90, axis=(0, 1, 3), keepdims=True)
            assert_equal(res.shape, (1, 1, 7, 1))

    @pytest.mark.parametrize('q', [7, [1, 7]])
    @pytest.mark.parametrize(argnames='axis', argvalues=[None, 1, (1,), (0, 1), (-3, -1)])
    @pytest.mark.filterwarnings('ignore:All-NaN slice:RuntimeWarning')
    def test_keepdims_out(self, q, axis):
        d = np.ones((3, 5, 7, 11))
        w = np.random.random((4, 200)) * np.array(d.shape)[:, None]
        w = w.astype(np.intp)
        d[tuple(w)] = np.nan
        if axis is None:
            shape_out = (1,) * d.ndim
        else:
            axis_norm = normalize_axis_tuple(axis, d.ndim)
            shape_out = tuple((1 if i in axis_norm else d.shape[i] for i in range(d.ndim)))
        shape_out = np.shape(q) + shape_out
        out = np.empty(shape_out)
        result = np.nanpercentile(d, q, axis=axis, keepdims=True, out=out)
        assert result is out
        assert_equal(result.shape, shape_out)

    def test_out(self):
        mat = np.random.rand(3, 3)
        nan_mat = np.insert(mat, [0, 2], np.nan, axis=1)
        resout = np.zeros(3)
        tgt = np.percentile(mat, 42, axis=1)
        res = np.nanpercentile(nan_mat, 42, axis=1, out=resout)
        assert_almost_equal(res, resout)
        assert_almost_equal(res, tgt)
        resout = np.zeros(())
        tgt = np.percentile(mat, 42, axis=None)
        res = np.nanpercentile(nan_mat, 42, axis=None, out=resout)
        assert_almost_equal(res, resout)
        assert_almost_equal(res, tgt)
        res = np.nanpercentile(nan_mat, 42, axis=(0, 1), out=resout)
        assert_almost_equal(res, resout)
        assert_almost_equal(res, tgt)

    def test_complex(self):
        arr_c = np.array([0.5 + 3j, 2.1 + 0.5j, 1.6 + 2.3j], dtype='G')
        assert_raises(TypeError, np.nanpercentile, arr_c, 0.5)
        arr_c = np.array([0.5 + 3j, 2.1 + 0.5j, 1.6 + 2.3j], dtype='D')
        assert_raises(TypeError, np.nanpercentile, arr_c, 0.5)
        arr_c = np.array([0.5 + 3j, 2.1 + 0.5j, 1.6 + 2.3j], dtype='F')
        assert_raises(TypeError, np.nanpercentile, arr_c, 0.5)

    def test_result_values(self):
        tgt = [np.percentile(d, 28) for d in _rdat]
        res = np.nanpercentile(_ndat, 28, axis=1)
        assert_almost_equal(res, tgt)
        tgt = np.transpose([np.percentile(d, (28, 98)) for d in _rdat])
        res = np.nanpercentile(_ndat, (28, 98), axis=1)
        assert_almost_equal(res, tgt)

    @pytest.mark.parametrize('axis', [None, 0, 1])
    @pytest.mark.parametrize('dtype', np.typecodes['Float'])
    @pytest.mark.parametrize('array', [np.array(np.nan), np.full((3, 3), np.nan)], ids=['0d', '2d'])
    def test_allnans(self, axis, dtype, array):
        if axis is not None and array.ndim == 0:
            pytest.skip(f'`axis != None` not supported for 0d arrays')
        array = array.astype(dtype)
        with pytest.warns(RuntimeWarning, match='All-NaN slice encountered'):
            out = np.nanpercentile(array, 60, axis=axis)
        assert np.isnan(out).all()
        assert out.dtype == array.dtype

    def test_empty(self):
        mat = np.zeros((0, 3))
        for axis in [0, None]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                assert_(np.isnan(np.nanpercentile(mat, 40, axis=axis)).all())
                assert_(len(w) == 1)
                assert_(issubclass(w[0].category, RuntimeWarning))
        for axis in [1]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                assert_equal(np.nanpercentile(mat, 40, axis=axis), np.zeros([]))
                assert_(len(w) == 0)

    def test_scalar(self):
        assert_equal(np.nanpercentile(0.0, 100), 0.0)
        a = np.arange(6)
        r = np.nanpercentile(a, 50, axis=0)
        assert_equal(r, 2.5)
        assert_(np.isscalar(r))

    def test_extended_axis_invalid(self):
        d = np.ones((3, 5, 7, 11))
        assert_raises(np.AxisError, np.nanpercentile, d, q=5, axis=-5)
        assert_raises(np.AxisError, np.nanpercentile, d, q=5, axis=(0, -5))
        assert_raises(np.AxisError, np.nanpercentile, d, q=5, axis=4)
        assert_raises(np.AxisError, np.nanpercentile, d, q=5, axis=(0, 4))
        assert_raises(ValueError, np.nanpercentile, d, q=5, axis=(1, 1))

    def test_multiple_percentiles(self):
        perc = [50, 100]
        mat = np.ones((4, 3))
        nan_mat = np.nan * mat
        large_mat = np.ones((3, 4, 5))
        large_mat[:, 0:2:4, :] = 0
        large_mat[:, :, 3:] *= 2
        for axis in [None, 0, 1]:
            for keepdim in [False, True]:
                with suppress_warnings() as sup:
                    sup.filter(RuntimeWarning, 'All-NaN slice encountered')
                    val = np.percentile(mat, perc, axis=axis, keepdims=keepdim)
                    nan_val = np.nanpercentile(nan_mat, perc, axis=axis, keepdims=keepdim)
                    assert_equal(nan_val.shape, val.shape)
                    val = np.percentile(large_mat, perc, axis=axis, keepdims=keepdim)
                    nan_val = np.nanpercentile(large_mat, perc, axis=axis, keepdims=keepdim)
                    assert_equal(nan_val, val)
        megamat = np.ones((3, 4, 5, 6))
        assert_equal(np.nanpercentile(megamat, perc, axis=(1, 2)).shape, (2, 3, 6))