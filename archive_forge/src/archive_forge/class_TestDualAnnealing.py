from scipy.optimize import dual_annealing, Bounds
from scipy.optimize._dual_annealing import EnergyState
from scipy.optimize._dual_annealing import LocalSearchWrapper
from scipy.optimize._dual_annealing import ObjectiveFunWrapper
from scipy.optimize._dual_annealing import StrategyChain
from scipy.optimize._dual_annealing import VisitingDistribution
from scipy.optimize import rosen, rosen_der
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_array_less
from pytest import raises as assert_raises
from scipy._lib._util import check_random_state
class TestDualAnnealing:

    def setup_method(self):
        self.weirdfunc = lambda x: np.inf
        self.ld_bounds = [(-5.12, 5.12)] * 2
        self.hd_bounds = self.ld_bounds * 4
        self.nbtestvalues = 5000
        self.high_temperature = 5230
        self.low_temperature = 0.1
        self.qv = 2.62
        self.seed = 1234
        self.rs = check_random_state(self.seed)
        self.nb_fun_call = 0
        self.ngev = 0

    def callback(self, x, f, context):
        if f <= 1.0:
            return True

    def func(self, x, args=()):
        if args:
            shift = args
        else:
            shift = 0
        y = np.sum((x - shift) ** 2 - 10 * np.cos(2 * np.pi * (x - shift))) + 10 * np.size(x) + shift
        self.nb_fun_call += 1
        return y

    def rosen_der_wrapper(self, x, args=()):
        self.ngev += 1
        return rosen_der(x, *args)

    @pytest.mark.parametrize('qv', [1.1, 1.41, 2, 2.62, 2.9])
    def test_visiting_stepping(self, qv):
        lu = list(zip(*self.ld_bounds))
        lower = np.array(lu[0])
        upper = np.array(lu[1])
        dim = lower.size
        vd = VisitingDistribution(lower, upper, qv, self.rs)
        values = np.zeros(dim)
        x_step_low = vd.visiting(values, 0, self.high_temperature)
        assert_equal(np.not_equal(x_step_low, 0), True)
        values = np.zeros(dim)
        x_step_high = vd.visiting(values, dim, self.high_temperature)
        assert_equal(np.not_equal(x_step_high[0], 0), True)

    @pytest.mark.parametrize('qv', [2.25, 2.62, 2.9])
    def test_visiting_dist_high_temperature(self, qv):
        lu = list(zip(*self.ld_bounds))
        lower = np.array(lu[0])
        upper = np.array(lu[1])
        vd = VisitingDistribution(lower, upper, qv, self.rs)
        values = vd.visit_fn(self.high_temperature, self.nbtestvalues)
        assert_array_less(np.min(values), 1e-10)
        assert_array_less(10000000000.0, np.max(values))

    def test_reset(self):
        owf = ObjectiveFunWrapper(self.weirdfunc)
        lu = list(zip(*self.ld_bounds))
        lower = np.array(lu[0])
        upper = np.array(lu[1])
        es = EnergyState(lower, upper)
        assert_raises(ValueError, es.reset, owf, check_random_state(None))

    def test_low_dim(self):
        ret = dual_annealing(self.func, self.ld_bounds, seed=self.seed)
        assert_allclose(ret.fun, 0.0, atol=1e-12)
        assert ret.success

    def test_high_dim(self):
        ret = dual_annealing(self.func, self.hd_bounds, seed=self.seed)
        assert_allclose(ret.fun, 0.0, atol=1e-12)
        assert ret.success

    def test_low_dim_no_ls(self):
        ret = dual_annealing(self.func, self.ld_bounds, no_local_search=True, seed=self.seed)
        assert_allclose(ret.fun, 0.0, atol=0.0001)

    def test_high_dim_no_ls(self):
        ret = dual_annealing(self.func, self.hd_bounds, no_local_search=True, seed=self.seed)
        assert_allclose(ret.fun, 0.0, atol=0.0001)

    def test_nb_fun_call(self):
        ret = dual_annealing(self.func, self.ld_bounds, seed=self.seed)
        assert_equal(self.nb_fun_call, ret.nfev)

    def test_nb_fun_call_no_ls(self):
        ret = dual_annealing(self.func, self.ld_bounds, no_local_search=True, seed=self.seed)
        assert_equal(self.nb_fun_call, ret.nfev)

    def test_max_reinit(self):
        assert_raises(ValueError, dual_annealing, self.weirdfunc, self.ld_bounds)

    def test_reproduce(self):
        res1 = dual_annealing(self.func, self.ld_bounds, seed=self.seed)
        res2 = dual_annealing(self.func, self.ld_bounds, seed=self.seed)
        res3 = dual_annealing(self.func, self.ld_bounds, seed=self.seed)
        assert_equal(res1.x, res2.x)
        assert_equal(res1.x, res3.x)

    def test_rand_gen(self):
        rng = np.random.default_rng(1)
        res1 = dual_annealing(self.func, self.ld_bounds, seed=rng)
        rng = np.random.default_rng(1)
        res2 = dual_annealing(self.func, self.ld_bounds, seed=rng)
        assert_equal(res1.x, res2.x)

    def test_bounds_integrity(self):
        wrong_bounds = [(-5.12, 5.12), (1, 0), (5.12, 5.12)]
        assert_raises(ValueError, dual_annealing, self.func, wrong_bounds)

    def test_bound_validity(self):
        invalid_bounds = [(-5, 5), (-np.inf, 0), (-5, 5)]
        assert_raises(ValueError, dual_annealing, self.func, invalid_bounds)
        invalid_bounds = [(-5, 5), (0, np.inf), (-5, 5)]
        assert_raises(ValueError, dual_annealing, self.func, invalid_bounds)
        invalid_bounds = [(-5, 5), (0, np.nan), (-5, 5)]
        assert_raises(ValueError, dual_annealing, self.func, invalid_bounds)

    def test_deprecated_local_search_options_bounds(self):

        def func(x):
            return np.sum((x - 5) * (x - 1))
        bounds = list(zip([-6, -5], [6, 5]))
        with pytest.warns(RuntimeWarning, match='Method CG cannot handle '):
            dual_annealing(func, bounds=bounds, minimizer_kwargs={'method': 'CG', 'bounds': bounds})

    def test_minimizer_kwargs_bounds(self):

        def func(x):
            return np.sum((x - 5) * (x - 1))
        bounds = list(zip([-6, -5], [6, 5]))
        dual_annealing(func, bounds=bounds, minimizer_kwargs={'method': 'SLSQP', 'bounds': bounds})
        with pytest.warns(RuntimeWarning, match='Method CG cannot handle '):
            dual_annealing(func, bounds=bounds, minimizer_kwargs={'method': 'CG', 'bounds': bounds})

    def test_max_fun_ls(self):
        ret = dual_annealing(self.func, self.ld_bounds, maxfun=100, seed=self.seed)
        ls_max_iter = min(max(len(self.ld_bounds) * LocalSearchWrapper.LS_MAXITER_RATIO, LocalSearchWrapper.LS_MAXITER_MIN), LocalSearchWrapper.LS_MAXITER_MAX)
        assert ret.nfev <= 100 + ls_max_iter
        assert not ret.success

    def test_max_fun_no_ls(self):
        ret = dual_annealing(self.func, self.ld_bounds, no_local_search=True, maxfun=500, seed=self.seed)
        assert ret.nfev <= 500
        assert not ret.success

    def test_maxiter(self):
        ret = dual_annealing(self.func, self.ld_bounds, maxiter=700, seed=self.seed)
        assert ret.nit <= 700

    def test_fun_args_ls(self):
        ret = dual_annealing(self.func, self.ld_bounds, args=(3.14159,), seed=self.seed)
        assert_allclose(ret.fun, 3.14159, atol=1e-06)

    def test_fun_args_no_ls(self):
        ret = dual_annealing(self.func, self.ld_bounds, args=(3.14159,), no_local_search=True, seed=self.seed)
        assert_allclose(ret.fun, 3.14159, atol=0.0001)

    def test_callback_stop(self):
        ret = dual_annealing(self.func, self.ld_bounds, callback=self.callback, seed=self.seed)
        assert ret.fun <= 1.0
        assert 'stop early' in ret.message[0]
        assert not ret.success

    @pytest.mark.parametrize('method, atol', [('Nelder-Mead', 2e-05), ('COBYLA', 1e-05), ('Powell', 1e-08), ('CG', 1e-08), ('BFGS', 1e-08), ('TNC', 1e-08), ('SLSQP', 2e-07)])
    def test_multi_ls_minimizer(self, method, atol):
        ret = dual_annealing(self.func, self.ld_bounds, minimizer_kwargs=dict(method=method), seed=self.seed)
        assert_allclose(ret.fun, 0.0, atol=atol)

    def test_wrong_restart_temp(self):
        assert_raises(ValueError, dual_annealing, self.func, self.ld_bounds, restart_temp_ratio=1)
        assert_raises(ValueError, dual_annealing, self.func, self.ld_bounds, restart_temp_ratio=0)

    def test_gradient_gnev(self):
        minimizer_opts = {'jac': self.rosen_der_wrapper}
        ret = dual_annealing(rosen, self.ld_bounds, minimizer_kwargs=minimizer_opts, seed=self.seed)
        assert ret.njev == self.ngev

    def test_from_docstring(self):

        def func(x):
            return np.sum(x * x - 10 * np.cos(2 * np.pi * x)) + 10 * np.size(x)
        lw = [-5.12] * 10
        up = [5.12] * 10
        ret = dual_annealing(func, bounds=list(zip(lw, up)), seed=1234)
        assert_allclose(ret.x, [-4.26437714e-09, -3.91699361e-09, -1.86149218e-09, -3.9716572e-09, -6.29151648e-09, -6.53145322e-09, -3.93616815e-09, -6.55623025e-09, -6.0577528e-09, -5.00668935e-09], atol=4e-08)
        assert_allclose(ret.fun, 0.0, atol=5e-13)

    @pytest.mark.parametrize('new_e, temp_step, accepted, accept_rate', [(0, 100, 1000, 1.0097587941791923), (0, 2, 1000, 1.2599210498948732), (10, 100, 878, 0.8786035869128718), (10, 60, 695, 0.6812920690579612), (2, 100, 990, 0.9897404249173424)])
    def test_accept_reject_probabilistic(self, new_e, temp_step, accepted, accept_rate):
        rs = check_random_state(123)
        count_accepted = 0
        iterations = 1000
        accept_param = -5
        current_energy = 1
        for _ in range(iterations):
            energy_state = EnergyState(lower=None, upper=None)
            energy_state.update_current(current_energy, [0])
            chain = StrategyChain(accept_param, None, None, None, rs, energy_state)
            chain.temperature_step = temp_step
            chain.accept_reject(j=1, e=new_e, x_visit=[2])
            if energy_state.current_energy == new_e:
                count_accepted += 1
        assert count_accepted == accepted
        pqv = 1 - (1 - accept_param) * (new_e - current_energy) / temp_step
        rate = 0 if pqv <= 0 else np.exp(np.log(pqv) / (1 - accept_param))
        assert_allclose(rate, accept_rate)

    def test_bounds_class(self):

        def func(x):
            f = np.sum(x * x - 10 * np.cos(2 * np.pi * x)) + 10 * np.size(x)
            return f
        lw = [-5.12] * 5
        up = [5.12] * 5
        up[0] = -2.0
        up[1] = -1.0
        lw[3] = 1.0
        lw[4] = 2.0
        bounds = Bounds(lw, up)
        ret_bounds_class = dual_annealing(func, bounds=bounds, seed=1234)
        bounds_old = list(zip(lw, up))
        ret_bounds_list = dual_annealing(func, bounds=bounds_old, seed=1234)
        assert_allclose(ret_bounds_class.x, ret_bounds_list.x, atol=1e-08)
        assert_allclose(ret_bounds_class.x, np.arange(-2, 3), atol=1e-07)
        assert_allclose(ret_bounds_list.fun, ret_bounds_class.fun, atol=1e-09)
        assert ret_bounds_list.nfev == ret_bounds_class.nfev

    def test_callable_jac_with_args_gh11052(self):
        rng = np.random.default_rng(94253637693657847462)

        def f(x, power):
            return np.sum(np.exp(x ** power))

        def jac(x, power):
            return np.exp(x ** power) * power * x ** (power - 1)
        res1 = dual_annealing(f, args=(2,), bounds=[[0, 1], [0, 1]], seed=rng, minimizer_kwargs=dict(method='L-BFGS-B'))
        res2 = dual_annealing(f, args=(2,), bounds=[[0, 1], [0, 1]], seed=rng, minimizer_kwargs=dict(method='L-BFGS-B', jac=jac))
        assert_allclose(res1.fun, res2.fun, rtol=1e-06)