import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
class _TestRBFInterpolator:

    @pytest.mark.parametrize('kernel', sorted(_SCALE_INVARIANT))
    def test_scale_invariance_1d(self, kernel):
        seq = Halton(1, scramble=False, seed=np.random.RandomState())
        x = 3 * seq.random(50)
        y = _1d_test_function(x)
        xitp = 3 * seq.random(50)
        yitp1 = self.build(x, y, epsilon=1.0, kernel=kernel)(xitp)
        yitp2 = self.build(x, y, epsilon=2.0, kernel=kernel)(xitp)
        assert_allclose(yitp1, yitp2, atol=1e-08)

    @pytest.mark.parametrize('kernel', sorted(_SCALE_INVARIANT))
    def test_scale_invariance_2d(self, kernel):
        seq = Halton(2, scramble=False, seed=np.random.RandomState())
        x = seq.random(100)
        y = _2d_test_function(x)
        xitp = seq.random(100)
        yitp1 = self.build(x, y, epsilon=1.0, kernel=kernel)(xitp)
        yitp2 = self.build(x, y, epsilon=2.0, kernel=kernel)(xitp)
        assert_allclose(yitp1, yitp2, atol=1e-08)

    @pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
    def test_extreme_domains(self, kernel):
        seq = Halton(2, scramble=False, seed=np.random.RandomState())
        scale = 1e+50
        shift = 1e+55
        x = seq.random(100)
        y = _2d_test_function(x)
        xitp = seq.random(100)
        if kernel in _SCALE_INVARIANT:
            yitp1 = self.build(x, y, kernel=kernel)(xitp)
            yitp2 = self.build(x * scale + shift, y, kernel=kernel)(xitp * scale + shift)
        else:
            yitp1 = self.build(x, y, epsilon=5.0, kernel=kernel)(xitp)
            yitp2 = self.build(x * scale + shift, y, epsilon=5.0 / scale, kernel=kernel)(xitp * scale + shift)
        assert_allclose(yitp1, yitp2, atol=1e-08)

    def test_polynomial_reproduction(self):
        rng = np.random.RandomState(0)
        seq = Halton(2, scramble=False, seed=rng)
        degree = 3
        x = seq.random(50)
        xitp = seq.random(50)
        P = _vandermonde(x, degree)
        Pitp = _vandermonde(xitp, degree)
        poly_coeffs = rng.normal(0.0, 1.0, P.shape[1])
        y = P.dot(poly_coeffs)
        yitp1 = Pitp.dot(poly_coeffs)
        yitp2 = self.build(x, y, degree=degree)(xitp)
        assert_allclose(yitp1, yitp2, atol=1e-08)

    @pytest.mark.slow
    def test_chunking(self, monkeypatch):
        rng = np.random.RandomState(0)
        seq = Halton(2, scramble=False, seed=rng)
        degree = 3
        largeN = 1000 + 33
        x = seq.random(50)
        xitp = seq.random(largeN)
        P = _vandermonde(x, degree)
        Pitp = _vandermonde(xitp, degree)
        poly_coeffs = rng.normal(0.0, 1.0, P.shape[1])
        y = P.dot(poly_coeffs)
        yitp1 = Pitp.dot(poly_coeffs)
        interp = self.build(x, y, degree=degree)
        ce_real = interp._chunk_evaluator

        def _chunk_evaluator(*args, **kwargs):
            kwargs.update(memory_budget=100)
            return ce_real(*args, **kwargs)
        monkeypatch.setattr(interp, '_chunk_evaluator', _chunk_evaluator)
        yitp2 = interp(xitp)
        assert_allclose(yitp1, yitp2, atol=1e-08)

    def test_vector_data(self):
        seq = Halton(2, scramble=False, seed=np.random.RandomState())
        x = seq.random(100)
        xitp = seq.random(100)
        y = np.array([_2d_test_function(x), _2d_test_function(x[:, ::-1])]).T
        yitp1 = self.build(x, y)(xitp)
        yitp2 = self.build(x, y[:, 0])(xitp)
        yitp3 = self.build(x, y[:, 1])(xitp)
        assert_allclose(yitp1[:, 0], yitp2)
        assert_allclose(yitp1[:, 1], yitp3)

    def test_complex_data(self):
        seq = Halton(2, scramble=False, seed=np.random.RandomState())
        x = seq.random(100)
        xitp = seq.random(100)
        y = _2d_test_function(x) + 1j * _2d_test_function(x[:, ::-1])
        yitp1 = self.build(x, y)(xitp)
        yitp2 = self.build(x, y.real)(xitp)
        yitp3 = self.build(x, y.imag)(xitp)
        assert_allclose(yitp1.real, yitp2)
        assert_allclose(yitp1.imag, yitp3)

    @pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
    def test_interpolation_misfit_1d(self, kernel):
        seq = Halton(1, scramble=False, seed=np.random.RandomState())
        x = 3 * seq.random(50)
        xitp = 3 * seq.random(50)
        y = _1d_test_function(x)
        ytrue = _1d_test_function(xitp)
        yitp = self.build(x, y, epsilon=5.0, kernel=kernel)(xitp)
        mse = np.mean((yitp - ytrue) ** 2)
        assert mse < 0.0001

    @pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
    def test_interpolation_misfit_2d(self, kernel):
        seq = Halton(2, scramble=False, seed=np.random.RandomState())
        x = seq.random(100)
        xitp = seq.random(100)
        y = _2d_test_function(x)
        ytrue = _2d_test_function(xitp)
        yitp = self.build(x, y, epsilon=5.0, kernel=kernel)(xitp)
        mse = np.mean((yitp - ytrue) ** 2)
        assert mse < 0.0002

    @pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
    def test_smoothing_misfit(self, kernel):
        rng = np.random.RandomState(0)
        seq = Halton(1, scramble=False, seed=rng)
        noise = 0.2
        rmse_tol = 0.1
        smoothing_range = 10 ** np.linspace(-4, 1, 20)
        x = 3 * seq.random(100)
        y = _1d_test_function(x) + rng.normal(0.0, noise, (100,))
        ytrue = _1d_test_function(x)
        rmse_within_tol = False
        for smoothing in smoothing_range:
            ysmooth = self.build(x, y, epsilon=1.0, smoothing=smoothing, kernel=kernel)(x)
            rmse = np.sqrt(np.mean((ysmooth - ytrue) ** 2))
            if rmse < rmse_tol:
                rmse_within_tol = True
                break
        assert rmse_within_tol

    def test_array_smoothing(self):
        rng = np.random.RandomState(0)
        seq = Halton(1, scramble=False, seed=rng)
        degree = 2
        x = seq.random(50)
        P = _vandermonde(x, degree)
        poly_coeffs = rng.normal(0.0, 1.0, P.shape[1])
        y = P.dot(poly_coeffs)
        y_with_outlier = np.copy(y)
        y_with_outlier[10] += 1.0
        smoothing = np.zeros((50,))
        smoothing[10] = 1000.0
        yitp = self.build(x, y_with_outlier, smoothing=smoothing)(x)
        assert_allclose(yitp, y, atol=0.0001)

    def test_inconsistent_x_dimensions_error(self):
        y = Halton(2, scramble=False, seed=np.random.RandomState()).random(10)
        d = _2d_test_function(y)
        x = Halton(1, scramble=False, seed=np.random.RandomState()).random(10)
        match = 'Expected the second axis of `x`'
        with pytest.raises(ValueError, match=match):
            self.build(y, d)(x)

    def test_inconsistent_d_length_error(self):
        y = np.linspace(0, 1, 5)[:, None]
        d = np.zeros(1)
        match = 'Expected the first axis of `d`'
        with pytest.raises(ValueError, match=match):
            self.build(y, d)

    def test_y_not_2d_error(self):
        y = np.linspace(0, 1, 5)
        d = np.zeros(5)
        match = '`y` must be a 2-dimensional array.'
        with pytest.raises(ValueError, match=match):
            self.build(y, d)

    def test_inconsistent_smoothing_length_error(self):
        y = np.linspace(0, 1, 5)[:, None]
        d = np.zeros(5)
        smoothing = np.ones(1)
        match = 'Expected `smoothing` to be'
        with pytest.raises(ValueError, match=match):
            self.build(y, d, smoothing=smoothing)

    def test_invalid_kernel_name_error(self):
        y = np.linspace(0, 1, 5)[:, None]
        d = np.zeros(5)
        match = '`kernel` must be one of'
        with pytest.raises(ValueError, match=match):
            self.build(y, d, kernel='test')

    def test_epsilon_not_specified_error(self):
        y = np.linspace(0, 1, 5)[:, None]
        d = np.zeros(5)
        for kernel in _AVAILABLE:
            if kernel in _SCALE_INVARIANT:
                continue
            match = '`epsilon` must be specified'
            with pytest.raises(ValueError, match=match):
                self.build(y, d, kernel=kernel)

    def test_x_not_2d_error(self):
        y = np.linspace(0, 1, 5)[:, None]
        x = np.linspace(0, 1, 5)
        d = np.zeros(5)
        match = '`x` must be a 2-dimensional array.'
        with pytest.raises(ValueError, match=match):
            self.build(y, d)(x)

    def test_not_enough_observations_error(self):
        y = np.linspace(0, 1, 1)[:, None]
        d = np.zeros(1)
        match = 'At least 2 data points are required'
        with pytest.raises(ValueError, match=match):
            self.build(y, d, kernel='thin_plate_spline')

    def test_degree_warning(self):
        y = np.linspace(0, 1, 5)[:, None]
        d = np.zeros(5)
        for kernel, deg in _NAME_TO_MIN_DEGREE.items():
            match = f'`degree` should not be below {deg}'
            with pytest.warns(Warning, match=match):
                self.build(y, d, epsilon=1.0, kernel=kernel, degree=deg - 1)

    def test_rank_error(self):
        y = np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        d = np.array([0.0, 0.0, 0.0])
        match = 'does not have full column rank'
        with pytest.raises(LinAlgError, match=match):
            self.build(y, d, kernel='thin_plate_spline')(y)

    def test_single_point(self):
        for dim in [1, 2, 3]:
            y = np.zeros((1, dim))
            d = np.ones((1,))
            f = self.build(y, d, kernel='linear')(y)
            assert_allclose(d, f)

    def test_pickleable(self):
        seq = Halton(1, scramble=False, seed=np.random.RandomState(2305982309))
        x = 3 * seq.random(50)
        xitp = 3 * seq.random(50)
        y = _1d_test_function(x)
        interp = self.build(x, y)
        yitp1 = interp(xitp)
        yitp2 = pickle.loads(pickle.dumps(interp))(xitp)
        assert_array_equal(yitp1, yitp2)