import pickle
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx
from scipy.optimize import (
from scipy.special import logsumexp
from sklearn._loss.link import IdentityLink, _inclusive_low_high
from sklearn._loss.loss import (
from sklearn.utils import _IS_WASM, assert_all_finite
from sklearn.utils._testing import create_memmap_backed_data, skip_if_32bit
@pytest.mark.parametrize('loss, func, random_dist', [(HalfSquaredError(), np.mean, 'normal'), (AbsoluteError(), np.median, 'normal'), (PinballLoss(quantile=0.25), lambda x: np.percentile(x, q=25), 'normal'), (HalfPoissonLoss(), np.mean, 'poisson'), (HalfGammaLoss(), np.mean, 'exponential'), (HalfTweedieLoss(), np.mean, 'exponential'), (HalfBinomialLoss(), np.mean, 'binomial')])
def test_specific_fit_intercept_only(loss, func, random_dist, global_random_seed):
    """Test that fit_intercept_only returns the correct functional.

    We test the functional for specific, meaningful distributions, e.g.
    squared error estimates the expectation of a probability distribution.
    """
    rng = np.random.RandomState(global_random_seed)
    if random_dist == 'binomial':
        y_train = rng.binomial(1, 0.5, size=100)
    else:
        y_train = getattr(rng, random_dist)(size=100)
    baseline_prediction = loss.fit_intercept_only(y_true=y_train)
    assert_all_finite(baseline_prediction)
    assert baseline_prediction == approx(loss.link.link(func(y_train)))
    assert loss.link.inverse(baseline_prediction) == approx(func(y_train))
    if isinstance(loss, IdentityLink):
        assert_allclose(loss.link.inverse(baseline_prediction), baseline_prediction)
    if loss.interval_y_true.low_inclusive:
        y_train.fill(loss.interval_y_true.low)
        baseline_prediction = loss.fit_intercept_only(y_true=y_train)
        assert_all_finite(baseline_prediction)
    if loss.interval_y_true.high_inclusive:
        y_train.fill(loss.interval_y_true.high)
        baseline_prediction = loss.fit_intercept_only(y_true=y_train)
        assert_all_finite(baseline_prediction)