from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
class TestnanopsDataFrame:

    def setup_method(self):
        nanops._USE_BOTTLENECK = False
        arr_shape = (11, 7)
        self.arr_float = np.random.default_rng(2).standard_normal(arr_shape)
        self.arr_float1 = np.random.default_rng(2).standard_normal(arr_shape)
        self.arr_complex = self.arr_float + self.arr_float1 * 1j
        self.arr_int = np.random.default_rng(2).integers(-10, 10, arr_shape)
        self.arr_bool = np.random.default_rng(2).integers(0, 2, arr_shape) == 0
        self.arr_str = np.abs(self.arr_float).astype('S')
        self.arr_utf = np.abs(self.arr_float).astype('U')
        self.arr_date = np.random.default_rng(2).integers(0, 20000, arr_shape).astype('M8[ns]')
        self.arr_tdelta = np.random.default_rng(2).integers(0, 20000, arr_shape).astype('m8[ns]')
        self.arr_nan = np.tile(np.nan, arr_shape)
        self.arr_float_nan = np.vstack([self.arr_float, self.arr_nan])
        self.arr_float1_nan = np.vstack([self.arr_float1, self.arr_nan])
        self.arr_nan_float1 = np.vstack([self.arr_nan, self.arr_float1])
        self.arr_nan_nan = np.vstack([self.arr_nan, self.arr_nan])
        self.arr_inf = self.arr_float * np.inf
        self.arr_float_inf = np.vstack([self.arr_float, self.arr_inf])
        self.arr_nan_inf = np.vstack([self.arr_nan, self.arr_inf])
        self.arr_float_nan_inf = np.vstack([self.arr_float, self.arr_nan, self.arr_inf])
        self.arr_nan_nan_inf = np.vstack([self.arr_nan, self.arr_nan, self.arr_inf])
        self.arr_obj = np.vstack([self.arr_float.astype('O'), self.arr_int.astype('O'), self.arr_bool.astype('O'), self.arr_complex.astype('O'), self.arr_str.astype('O'), self.arr_utf.astype('O'), self.arr_date.astype('O'), self.arr_tdelta.astype('O')])
        with np.errstate(invalid='ignore'):
            self.arr_nan_nanj = self.arr_nan + self.arr_nan * 1j
            self.arr_complex_nan = np.vstack([self.arr_complex, self.arr_nan_nanj])
            self.arr_nan_infj = self.arr_inf * 1j
            self.arr_complex_nan_infj = np.vstack([self.arr_complex, self.arr_nan_infj])
        self.arr_float_2d = self.arr_float
        self.arr_float1_2d = self.arr_float1
        self.arr_nan_2d = self.arr_nan
        self.arr_float_nan_2d = self.arr_float_nan
        self.arr_float1_nan_2d = self.arr_float1_nan
        self.arr_nan_float1_2d = self.arr_nan_float1
        self.arr_float_1d = self.arr_float[:, 0]
        self.arr_float1_1d = self.arr_float1[:, 0]
        self.arr_nan_1d = self.arr_nan[:, 0]
        self.arr_float_nan_1d = self.arr_float_nan[:, 0]
        self.arr_float1_nan_1d = self.arr_float1_nan[:, 0]
        self.arr_nan_float1_1d = self.arr_nan_float1[:, 0]

    def teardown_method(self):
        nanops._USE_BOTTLENECK = use_bn

    def check_results(self, targ, res, axis, check_dtype=True):
        res = getattr(res, 'asm8', res)
        if axis != 0 and hasattr(targ, 'shape') and targ.ndim and (targ.shape != res.shape):
            res = np.split(res, [targ.shape[0]], axis=0)[0]
        try:
            tm.assert_almost_equal(targ, res, check_dtype=check_dtype)
        except AssertionError:
            if hasattr(targ, 'dtype') and targ.dtype == 'm8[ns]':
                raise
            if not hasattr(res, 'dtype') or res.dtype.kind not in ['c', 'O']:
                raise
            if res.dtype.kind == 'O':
                if targ.dtype.kind != 'O':
                    res = res.astype(targ.dtype)
                else:
                    cast_dtype = 'c16' if hasattr(np, 'complex128') else 'f8'
                    res = res.astype(cast_dtype)
                    targ = targ.astype(cast_dtype)
            elif targ.dtype.kind == 'O':
                raise
            tm.assert_almost_equal(np.real(targ), np.real(res), check_dtype=check_dtype)
            tm.assert_almost_equal(np.imag(targ), np.imag(res), check_dtype=check_dtype)

    def check_fun_data(self, testfunc, targfunc, testarval, targarval, skipna, check_dtype=True, empty_targfunc=None, **kwargs):
        for axis in list(range(targarval.ndim)) + [None]:
            targartempval = targarval if skipna else testarval
            if skipna and empty_targfunc and isna(targartempval).all():
                targ = empty_targfunc(targartempval, axis=axis, **kwargs)
            else:
                targ = targfunc(targartempval, axis=axis, **kwargs)
            if targartempval.dtype == object and (targfunc is np.any or targfunc is np.all):
                if isinstance(targ, np.ndarray):
                    targ = targ.astype(bool)
                else:
                    targ = bool(targ)
            res = testfunc(testarval, axis=axis, skipna=skipna, **kwargs)
            if isinstance(targ, np.complex128) and isinstance(res, float) and np.isnan(targ) and np.isnan(res):
                targ = res
            self.check_results(targ, res, axis, check_dtype=check_dtype)
            if skipna:
                res = testfunc(testarval, axis=axis, **kwargs)
                self.check_results(targ, res, axis, check_dtype=check_dtype)
            if axis is None:
                res = testfunc(testarval, skipna=skipna, **kwargs)
                self.check_results(targ, res, axis, check_dtype=check_dtype)
            if skipna and axis is None:
                res = testfunc(testarval, **kwargs)
                self.check_results(targ, res, axis, check_dtype=check_dtype)
        if testarval.ndim <= 1:
            return
        testarval2 = np.take(testarval, 0, axis=-1)
        targarval2 = np.take(targarval, 0, axis=-1)
        self.check_fun_data(testfunc, targfunc, testarval2, targarval2, skipna=skipna, check_dtype=check_dtype, empty_targfunc=empty_targfunc, **kwargs)

    def check_fun(self, testfunc, targfunc, testar, skipna, empty_targfunc=None, **kwargs):
        targar = testar
        if testar.endswith('_nan') and hasattr(self, testar[:-4]):
            targar = testar[:-4]
        testarval = getattr(self, testar)
        targarval = getattr(self, targar)
        self.check_fun_data(testfunc, targfunc, testarval, targarval, skipna=skipna, empty_targfunc=empty_targfunc, **kwargs)

    def check_funs(self, testfunc, targfunc, skipna, allow_complex=True, allow_all_nan=True, allow_date=True, allow_tdelta=True, allow_obj=True, **kwargs):
        self.check_fun(testfunc, targfunc, 'arr_float', skipna, **kwargs)
        self.check_fun(testfunc, targfunc, 'arr_float_nan', skipna, **kwargs)
        self.check_fun(testfunc, targfunc, 'arr_int', skipna, **kwargs)
        self.check_fun(testfunc, targfunc, 'arr_bool', skipna, **kwargs)
        objs = [self.arr_float.astype('O'), self.arr_int.astype('O'), self.arr_bool.astype('O')]
        if allow_all_nan:
            self.check_fun(testfunc, targfunc, 'arr_nan', skipna, **kwargs)
        if allow_complex:
            self.check_fun(testfunc, targfunc, 'arr_complex', skipna, **kwargs)
            self.check_fun(testfunc, targfunc, 'arr_complex_nan', skipna, **kwargs)
            if allow_all_nan:
                self.check_fun(testfunc, targfunc, 'arr_nan_nanj', skipna, **kwargs)
            objs += [self.arr_complex.astype('O')]
        if allow_date:
            targfunc(self.arr_date)
            self.check_fun(testfunc, targfunc, 'arr_date', skipna, **kwargs)
            objs += [self.arr_date.astype('O')]
        if allow_tdelta:
            try:
                targfunc(self.arr_tdelta)
            except TypeError:
                pass
            else:
                self.check_fun(testfunc, targfunc, 'arr_tdelta', skipna, **kwargs)
                objs += [self.arr_tdelta.astype('O')]
        if allow_obj:
            self.arr_obj = np.vstack(objs)
            if allow_obj == 'convert':
                targfunc = partial(self._badobj_wrap, func=targfunc, allow_complex=allow_complex)
            self.check_fun(testfunc, targfunc, 'arr_obj', skipna, **kwargs)

    def _badobj_wrap(self, value, func, allow_complex=True, **kwargs):
        if value.dtype.kind == 'O':
            if allow_complex:
                value = value.astype('c16')
            else:
                value = value.astype('f8')
        return func(value, **kwargs)

    @pytest.mark.parametrize('nan_op,np_op', [(nanops.nanany, np.any), (nanops.nanall, np.all)])
    def test_nan_funcs(self, nan_op, np_op, skipna):
        self.check_funs(nan_op, np_op, skipna, allow_all_nan=False, allow_date=False)

    def test_nansum(self, skipna):
        self.check_funs(nanops.nansum, np.sum, skipna, allow_date=False, check_dtype=False, empty_targfunc=np.nansum)

    def test_nanmean(self, skipna):
        self.check_funs(nanops.nanmean, np.mean, skipna, allow_obj=False, allow_date=False)

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_nanmedian(self, skipna):
        self.check_funs(nanops.nanmedian, np.median, skipna, allow_complex=False, allow_date=False, allow_obj='convert')

    @pytest.mark.parametrize('ddof', range(3))
    def test_nanvar(self, ddof, skipna):
        self.check_funs(nanops.nanvar, np.var, skipna, allow_complex=False, allow_date=False, allow_obj='convert', ddof=ddof)

    @pytest.mark.parametrize('ddof', range(3))
    def test_nanstd(self, ddof, skipna):
        self.check_funs(nanops.nanstd, np.std, skipna, allow_complex=False, allow_date=False, allow_obj='convert', ddof=ddof)

    @pytest.mark.parametrize('ddof', range(3))
    def test_nansem(self, ddof, skipna):
        sp_stats = pytest.importorskip('scipy.stats')
        with np.errstate(invalid='ignore'):
            self.check_funs(nanops.nansem, sp_stats.sem, skipna, allow_complex=False, allow_date=False, allow_tdelta=False, allow_obj='convert', ddof=ddof)

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    @pytest.mark.parametrize('nan_op,np_op', [(nanops.nanmin, np.min), (nanops.nanmax, np.max)])
    def test_nanops_with_warnings(self, nan_op, np_op, skipna):
        self.check_funs(nan_op, np_op, skipna, allow_obj=False)

    def _argminmax_wrap(self, value, axis=None, func=None):
        res = func(value, axis)
        nans = np.min(value, axis)
        nullnan = isna(nans)
        if res.ndim:
            res[nullnan] = -1
        elif hasattr(nullnan, 'all') and nullnan.all() or (not hasattr(nullnan, 'all') and nullnan):
            res = -1
        return res

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_nanargmax(self, skipna):
        func = partial(self._argminmax_wrap, func=np.argmax)
        self.check_funs(nanops.nanargmax, func, skipna, allow_obj=False)

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_nanargmin(self, skipna):
        func = partial(self._argminmax_wrap, func=np.argmin)
        self.check_funs(nanops.nanargmin, func, skipna, allow_obj=False)

    def _skew_kurt_wrap(self, values, axis=None, func=None):
        if not isinstance(values.dtype.type, np.floating):
            values = values.astype('f8')
        result = func(values, axis=axis, bias=False)
        if isinstance(result, np.ndarray):
            result[np.max(values, axis=axis) == np.min(values, axis=axis)] = 0
            return result
        elif np.max(values) == np.min(values):
            return 0.0
        return result

    def test_nanskew(self, skipna):
        sp_stats = pytest.importorskip('scipy.stats')
        func = partial(self._skew_kurt_wrap, func=sp_stats.skew)
        with np.errstate(invalid='ignore'):
            self.check_funs(nanops.nanskew, func, skipna, allow_complex=False, allow_date=False, allow_tdelta=False)

    def test_nankurt(self, skipna):
        sp_stats = pytest.importorskip('scipy.stats')
        func1 = partial(sp_stats.kurtosis, fisher=True)
        func = partial(self._skew_kurt_wrap, func=func1)
        with np.errstate(invalid='ignore'):
            self.check_funs(nanops.nankurt, func, skipna, allow_complex=False, allow_date=False, allow_tdelta=False)

    def test_nanprod(self, skipna):
        self.check_funs(nanops.nanprod, np.prod, skipna, allow_date=False, allow_tdelta=False, empty_targfunc=np.nanprod)

    def check_nancorr_nancov_2d(self, checkfun, targ0, targ1, **kwargs):
        res00 = checkfun(self.arr_float_2d, self.arr_float1_2d, **kwargs)
        res01 = checkfun(self.arr_float_2d, self.arr_float1_2d, min_periods=len(self.arr_float_2d) - 1, **kwargs)
        tm.assert_almost_equal(targ0, res00)
        tm.assert_almost_equal(targ0, res01)
        res10 = checkfun(self.arr_float_nan_2d, self.arr_float1_nan_2d, **kwargs)
        res11 = checkfun(self.arr_float_nan_2d, self.arr_float1_nan_2d, min_periods=len(self.arr_float_2d) - 1, **kwargs)
        tm.assert_almost_equal(targ1, res10)
        tm.assert_almost_equal(targ1, res11)
        targ2 = np.nan
        res20 = checkfun(self.arr_nan_2d, self.arr_float1_2d, **kwargs)
        res21 = checkfun(self.arr_float_2d, self.arr_nan_2d, **kwargs)
        res22 = checkfun(self.arr_nan_2d, self.arr_nan_2d, **kwargs)
        res23 = checkfun(self.arr_float_nan_2d, self.arr_nan_float1_2d, **kwargs)
        res24 = checkfun(self.arr_float_nan_2d, self.arr_nan_float1_2d, min_periods=len(self.arr_float_2d) - 1, **kwargs)
        res25 = checkfun(self.arr_float_2d, self.arr_float1_2d, min_periods=len(self.arr_float_2d) + 1, **kwargs)
        tm.assert_almost_equal(targ2, res20)
        tm.assert_almost_equal(targ2, res21)
        tm.assert_almost_equal(targ2, res22)
        tm.assert_almost_equal(targ2, res23)
        tm.assert_almost_equal(targ2, res24)
        tm.assert_almost_equal(targ2, res25)

    def check_nancorr_nancov_1d(self, checkfun, targ0, targ1, **kwargs):
        res00 = checkfun(self.arr_float_1d, self.arr_float1_1d, **kwargs)
        res01 = checkfun(self.arr_float_1d, self.arr_float1_1d, min_periods=len(self.arr_float_1d) - 1, **kwargs)
        tm.assert_almost_equal(targ0, res00)
        tm.assert_almost_equal(targ0, res01)
        res10 = checkfun(self.arr_float_nan_1d, self.arr_float1_nan_1d, **kwargs)
        res11 = checkfun(self.arr_float_nan_1d, self.arr_float1_nan_1d, min_periods=len(self.arr_float_1d) - 1, **kwargs)
        tm.assert_almost_equal(targ1, res10)
        tm.assert_almost_equal(targ1, res11)
        targ2 = np.nan
        res20 = checkfun(self.arr_nan_1d, self.arr_float1_1d, **kwargs)
        res21 = checkfun(self.arr_float_1d, self.arr_nan_1d, **kwargs)
        res22 = checkfun(self.arr_nan_1d, self.arr_nan_1d, **kwargs)
        res23 = checkfun(self.arr_float_nan_1d, self.arr_nan_float1_1d, **kwargs)
        res24 = checkfun(self.arr_float_nan_1d, self.arr_nan_float1_1d, min_periods=len(self.arr_float_1d) - 1, **kwargs)
        res25 = checkfun(self.arr_float_1d, self.arr_float1_1d, min_periods=len(self.arr_float_1d) + 1, **kwargs)
        tm.assert_almost_equal(targ2, res20)
        tm.assert_almost_equal(targ2, res21)
        tm.assert_almost_equal(targ2, res22)
        tm.assert_almost_equal(targ2, res23)
        tm.assert_almost_equal(targ2, res24)
        tm.assert_almost_equal(targ2, res25)

    def test_nancorr(self):
        targ0 = np.corrcoef(self.arr_float_2d, self.arr_float1_2d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0, 1]
        self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1)
        targ0 = np.corrcoef(self.arr_float_1d, self.arr_float1_1d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0, 1]
        self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method='pearson')

    def test_nancorr_pearson(self):
        targ0 = np.corrcoef(self.arr_float_2d, self.arr_float1_2d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0, 1]
        self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1, method='pearson')
        targ0 = np.corrcoef(self.arr_float_1d, self.arr_float1_1d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0, 1]
        self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method='pearson')

    def test_nancorr_kendall(self):
        sp_stats = pytest.importorskip('scipy.stats')
        targ0 = sp_stats.kendalltau(self.arr_float_2d, self.arr_float1_2d)[0]
        targ1 = sp_stats.kendalltau(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0]
        self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1, method='kendall')
        targ0 = sp_stats.kendalltau(self.arr_float_1d, self.arr_float1_1d)[0]
        targ1 = sp_stats.kendalltau(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0]
        self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method='kendall')

    def test_nancorr_spearman(self):
        sp_stats = pytest.importorskip('scipy.stats')
        targ0 = sp_stats.spearmanr(self.arr_float_2d, self.arr_float1_2d)[0]
        targ1 = sp_stats.spearmanr(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0]
        self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1, method='spearman')
        targ0 = sp_stats.spearmanr(self.arr_float_1d, self.arr_float1_1d)[0]
        targ1 = sp_stats.spearmanr(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0]
        self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method='spearman')

    def test_invalid_method(self):
        pytest.importorskip('scipy')
        targ0 = np.corrcoef(self.arr_float_2d, self.arr_float1_2d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0, 1]
        msg = "Unknown method 'foo', expected one of 'kendall', 'spearman'"
        with pytest.raises(ValueError, match=msg):
            self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method='foo')

    def test_nancov(self):
        targ0 = np.cov(self.arr_float_2d, self.arr_float1_2d)[0, 1]
        targ1 = np.cov(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0, 1]
        self.check_nancorr_nancov_2d(nanops.nancov, targ0, targ1)
        targ0 = np.cov(self.arr_float_1d, self.arr_float1_1d)[0, 1]
        targ1 = np.cov(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0, 1]
        self.check_nancorr_nancov_1d(nanops.nancov, targ0, targ1)