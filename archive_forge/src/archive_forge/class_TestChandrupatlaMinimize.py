import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_less
from scipy import stats
import scipy.optimize._chandrupatla as _chandrupatla
from scipy.optimize._chandrupatla import _chandrupatla_minimize
from itertools import permutations
class TestChandrupatlaMinimize:

    def f(self, x, loc):
        dist = stats.norm()
        return -dist.pdf(x - loc)

    @pytest.mark.parametrize('loc', [0.6, np.linspace(-1.05, 1.05, 10)])
    def test_basic(self, loc):
        res = _chandrupatla_minimize(self.f, -5, 0, 5, args=(loc,))
        ref = loc
        np.testing.assert_allclose(res.x, ref, rtol=1e-06)
        np.testing.assert_allclose(res.fun, -stats.norm.pdf(0), atol=0, rtol=0)
        assert res.x.shape == np.shape(ref)

    @pytest.mark.parametrize('shape', [tuple(), (12,), (3, 4), (3, 2, 2)])
    def test_vectorization(self, shape):
        loc = np.linspace(-0.05, 1.05, 12).reshape(shape) if shape else 0.6
        args = (loc,)

        @np.vectorize
        def chandrupatla_single(loc_single):
            return _chandrupatla_minimize(self.f, -5, 0, 5, args=(loc_single,))

        def f(*args, **kwargs):
            f.f_evals += 1
            return self.f(*args, **kwargs)
        f.f_evals = 0
        res = _chandrupatla_minimize(f, -5, 0, 5, args=args)
        refs = chandrupatla_single(loc).ravel()
        ref_x = [ref.x for ref in refs]
        assert_allclose(res.x.ravel(), ref_x)
        assert_equal(res.x.shape, shape)
        ref_fun = [ref.fun for ref in refs]
        assert_allclose(res.fun.ravel(), ref_fun)
        assert_equal(res.fun.shape, shape)
        assert_equal(res.fun, self.f(res.x, *args))
        ref_success = [ref.success for ref in refs]
        assert_equal(res.success.ravel(), ref_success)
        assert_equal(res.success.shape, shape)
        assert np.issubdtype(res.success.dtype, np.bool_)
        ref_flag = [ref.status for ref in refs]
        assert_equal(res.status.ravel(), ref_flag)
        assert_equal(res.status.shape, shape)
        assert np.issubdtype(res.status.dtype, np.integer)
        ref_nfev = [ref.nfev for ref in refs]
        assert_equal(res.nfev.ravel(), ref_nfev)
        assert_equal(np.max(res.nfev), f.f_evals)
        assert_equal(res.nfev.shape, res.fun.shape)
        assert np.issubdtype(res.nfev.dtype, np.integer)
        ref_nit = [ref.nit for ref in refs]
        assert_equal(res.nit.ravel(), ref_nit)
        assert_equal(np.max(res.nit), f.f_evals - 3)
        assert_equal(res.nit.shape, res.fun.shape)
        assert np.issubdtype(res.nit.dtype, np.integer)
        ref_xl = [ref.xl for ref in refs]
        assert_allclose(res.xl.ravel(), ref_xl)
        assert_equal(res.xl.shape, shape)
        ref_xm = [ref.xm for ref in refs]
        assert_allclose(res.xm.ravel(), ref_xm)
        assert_equal(res.xm.shape, shape)
        ref_xr = [ref.xr for ref in refs]
        assert_allclose(res.xr.ravel(), ref_xr)
        assert_equal(res.xr.shape, shape)
        ref_fl = [ref.fl for ref in refs]
        assert_allclose(res.fl.ravel(), ref_fl)
        assert_equal(res.fl.shape, shape)
        assert_allclose(res.fl, self.f(res.xl, *args))
        ref_fm = [ref.fm for ref in refs]
        assert_allclose(res.fm.ravel(), ref_fm)
        assert_equal(res.fm.shape, shape)
        assert_allclose(res.fm, self.f(res.xm, *args))
        ref_fr = [ref.fr for ref in refs]
        assert_allclose(res.fr.ravel(), ref_fr)
        assert_equal(res.fr.shape, shape)
        assert_allclose(res.fr, self.f(res.xr, *args))

    def test_flags(self):

        def f(xs, js):
            funcs = [lambda x: (x - 2.5) ** 2, lambda x: x - 10, lambda x: (x - 2.5) ** 4, lambda x: np.nan]
            return [funcs[j](x) for x, j in zip(xs, js)]
        args = (np.arange(4, dtype=np.int64),)
        res = _chandrupatla_minimize(f, [0] * 4, [2] * 4, [np.pi] * 4, args=args, maxiter=10)
        ref_flags = np.array([_chandrupatla._ECONVERGED, _chandrupatla._ESIGNERR, _chandrupatla._ECONVERR, _chandrupatla._EVALUEERR])
        assert_equal(res.status, ref_flags)

    def test_convergence(self):
        rng = np.random.default_rng(2585255913088665241)
        p = rng.random(size=3)
        bracket = (-5, 0, 5)
        args = (p,)
        kwargs0 = dict(args=args, xatol=0, xrtol=0, fatol=0, frtol=0)
        kwargs = kwargs0.copy()
        kwargs['xatol'] = 0.001
        res1 = _chandrupatla_minimize(self.f, *bracket, **kwargs)
        j1 = abs(res1.xr - res1.xl)
        assert_array_less(j1, 4 * kwargs['xatol'])
        kwargs['xatol'] = 1e-06
        res2 = _chandrupatla_minimize(self.f, *bracket, **kwargs)
        j2 = abs(res2.xr - res2.xl)
        assert_array_less(j2, 4 * kwargs['xatol'])
        assert_array_less(j2, j1)
        kwargs = kwargs0.copy()
        kwargs['xrtol'] = 0.001
        res1 = _chandrupatla_minimize(self.f, *bracket, **kwargs)
        j1 = abs(res1.xr - res1.xl)
        assert_array_less(j1, 4 * kwargs['xrtol'] * abs(res1.x))
        kwargs['xrtol'] = 1e-06
        res2 = _chandrupatla_minimize(self.f, *bracket, **kwargs)
        j2 = abs(res2.xr - res2.xl)
        assert_array_less(j2, 4 * kwargs['xrtol'] * abs(res2.x))
        assert_array_less(j2, j1)
        kwargs = kwargs0.copy()
        kwargs['fatol'] = 0.001
        res1 = _chandrupatla_minimize(self.f, *bracket, **kwargs)
        h1 = abs(res1.fl - 2 * res1.fm + res1.fr)
        assert_array_less(h1, 2 * kwargs['fatol'])
        kwargs['fatol'] = 1e-06
        res2 = _chandrupatla_minimize(self.f, *bracket, **kwargs)
        h2 = abs(res2.fl - 2 * res2.fm + res2.fr)
        assert_array_less(h2, 2 * kwargs['fatol'])
        assert_array_less(h2, h1)
        kwargs = kwargs0.copy()
        kwargs['frtol'] = 0.001
        res1 = _chandrupatla_minimize(self.f, *bracket, **kwargs)
        h1 = abs(res1.fl - 2 * res1.fm + res1.fr)
        assert_array_less(h1, 2 * kwargs['frtol'] * abs(res1.fun))
        kwargs['frtol'] = 1e-06
        res2 = _chandrupatla_minimize(self.f, *bracket, **kwargs)
        h2 = abs(res2.fl - 2 * res2.fm + res2.fr)
        assert_array_less(h2, 2 * kwargs['frtol'] * abs(res2.fun))
        assert_array_less(h2, h1)

    def test_maxiter_callback(self):
        loc = 0.612814
        bracket = (-5, 0, 5)
        maxiter = 5
        res = _chandrupatla_minimize(self.f, *bracket, args=(loc,), maxiter=maxiter)
        assert not np.any(res.success)
        assert np.all(res.nfev == maxiter + 3)
        assert np.all(res.nit == maxiter)

        def callback(res):
            callback.iter += 1
            callback.res = res
            assert hasattr(res, 'x')
            if callback.iter == 0:
                assert (res.xl, res.xm, res.xr) == bracket
            else:
                changed_xr = (res.xl == callback.xl) & (res.xr != callback.xr)
                changed_xl = (res.xl != callback.xl) & (res.xr == callback.xr)
                assert np.all(changed_xr | changed_xl)
            callback.xl = res.xl
            callback.xr = res.xr
            assert res.status == _chandrupatla._EINPROGRESS
            assert_equal(self.f(res.xl, loc), res.fl)
            assert_equal(self.f(res.xm, loc), res.fm)
            assert_equal(self.f(res.xr, loc), res.fr)
            assert_equal(self.f(res.x, loc), res.fun)
            if callback.iter == maxiter:
                raise StopIteration
        callback.xl = np.nan
        callback.xr = np.nan
        callback.iter = -1
        callback.res = None
        res2 = _chandrupatla_minimize(self.f, *bracket, args=(loc,), callback=callback)
        for key in res.keys():
            if key == 'status':
                assert res[key] == _chandrupatla._ECONVERR
                assert callback.res[key] == _chandrupatla._EINPROGRESS
                assert res2[key] == _chandrupatla._ECALLBACK
            else:
                assert res2[key] == callback.res[key] == res[key]

    @pytest.mark.parametrize('case', cases)
    def test_nit_expected(self, case):
        func, x1, nit = case
        step = 0.2
        x2 = x1 + step
        x1, x2, x3, f1, f2, f3 = _bracket_minimum(func, x1, x2)
        xatol = 0.0001
        fatol = 1e-06
        xrtol = 1e-16
        frtol = 1e-16
        res = _chandrupatla_minimize(func, x1, x2, x3, xatol=xatol, fatol=fatol, xrtol=xrtol, frtol=frtol)
        assert_equal(res.nit, nit)

    @pytest.mark.parametrize('loc', (0.65, [0.65, 0.7]))
    @pytest.mark.parametrize('dtype', (np.float16, np.float32, np.float64))
    def test_dtype(self, loc, dtype):
        loc = dtype(loc)

        def f(x, loc):
            assert x.dtype == dtype
            return ((x - loc) ** 2).astype(dtype)
        res = _chandrupatla_minimize(f, dtype(-3), dtype(1), dtype(5), args=(loc,))
        assert res.x.dtype == dtype
        assert_allclose(res.x, loc, rtol=np.sqrt(np.finfo(dtype).eps))

    def test_input_validation(self):
        message = '`func` must be callable.'
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(None, -4, 0, 4)
        message = 'Abscissae and function output must be real numbers.'
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(lambda x: x, -4 + 1j, 0, 4)
        message = 'shape mismatch: objects cannot be broadcast'
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(lambda x: x, [-2, -3], [0, 0], [3, 4, 5])
        message = 'The shape of the array returned by `func` must be the same'
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(lambda x: [x[0], x[1], x[1]], [-3, -3], [0, 0], [5, 5])
        message = 'Tolerances must be non-negative scalars.'
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(lambda x: x, -4, 0, 4, xatol=-1)
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(lambda x: x, -4, 0, 4, xrtol=np.nan)
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(lambda x: x, -4, 0, 4, fatol='ekki')
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(lambda x: x, -4, 0, 4, frtol=np.nan)
        message = '`maxiter` must be a non-negative integer.'
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(lambda x: x, -4, 0, 4, maxiter=1.5)
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(lambda x: x, -4, 0, 4, maxiter=-1)
        message = '`callback` must be callable.'
        with pytest.raises(ValueError, match=message):
            _chandrupatla_minimize(lambda x: x, -4, 0, 4, callback='shrubbery')

    def test_bracket_order(self):
        loc = np.linspace(-1, 1, 6)[:, np.newaxis]
        brackets = np.array(list(permutations([-5, 0, 5]))).T
        res = _chandrupatla_minimize(self.f, *brackets, args=(loc,))
        assert np.all(np.isclose(res.x, loc) | (res.fun == self.f(loc, loc)))
        ref = res.x[:, 0]
        assert_allclose(*np.broadcast_arrays(res.x.T, ref), rtol=1e-15)

    def test_special_cases(self):

        def f(x):
            assert np.issubdtype(x.dtype, np.floating)
            return (x - 1) ** 100
        with np.errstate(invalid='ignore'):
            res = _chandrupatla_minimize(f, -7, 0, 8, fatol=0, frtol=0)
        assert res.success
        assert_allclose(res.x, 1, rtol=0.001)
        assert_equal(res.fun, 0)

        def f(x):
            return (x - 1) ** 2
        res = _chandrupatla_minimize(f, 1, 1, 1)
        assert res.success
        assert_equal(res.x, 1)

        def f(x):
            return (x - 1) ** 2
        bracket = (-3, 1.1, 5)
        res = _chandrupatla_minimize(f, *bracket, maxiter=0)
        assert res.xl, res.xr == bracket
        assert res.nit == 0
        assert res.nfev == 3
        assert res.status == -2
        assert res.x == 1.1

        def f(x, c):
            return (x - c) ** 2 - 1
        res = _chandrupatla_minimize(f, -1, 0, 1, args=1 / 3)
        assert_allclose(res.x, 1 / 3)

        def f(x):
            return -np.sin(x)
        res = _chandrupatla_minimize(f, 0, 1, np.pi, xatol=0, xrtol=0, fatol=0, frtol=0)
        assert res.success
        assert res.xl < res.xm < res.xr
        assert f(res.xl) == f(res.xm) == f(res.xr)