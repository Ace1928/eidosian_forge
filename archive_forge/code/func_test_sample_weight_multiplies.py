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
@pytest.mark.parametrize('loss', LOSS_INSTANCES, ids=loss_instance_name)
@pytest.mark.parametrize('sample_weight', ['ones', 'random'])
def test_sample_weight_multiplies(loss, sample_weight, global_random_seed):
    """Test sample weights in loss, gradients and hessians.

    Make sure that passing sample weights to loss, gradient and hessian
    computation methods is equivalent to multiplying by the weights.
    """
    n_samples = 100
    y_true, raw_prediction = random_y_true_raw_prediction(loss=loss, n_samples=n_samples, y_bound=(-100, 100), raw_bound=(-5, 5), seed=global_random_seed)
    if sample_weight == 'ones':
        sample_weight = np.ones(shape=n_samples, dtype=np.float64)
    else:
        rng = np.random.RandomState(global_random_seed)
        sample_weight = rng.normal(size=n_samples).astype(np.float64)
    assert_allclose(loss.loss(y_true=y_true, raw_prediction=raw_prediction, sample_weight=sample_weight), sample_weight * loss.loss(y_true=y_true, raw_prediction=raw_prediction, sample_weight=None))
    losses, gradient = loss.loss_gradient(y_true=y_true, raw_prediction=raw_prediction, sample_weight=None)
    losses_sw, gradient_sw = loss.loss_gradient(y_true=y_true, raw_prediction=raw_prediction, sample_weight=sample_weight)
    assert_allclose(losses * sample_weight, losses_sw)
    if not loss.is_multiclass:
        assert_allclose(gradient * sample_weight, gradient_sw)
    else:
        assert_allclose(gradient * sample_weight[:, None], gradient_sw)
    gradient, hessian = loss.gradient_hessian(y_true=y_true, raw_prediction=raw_prediction, sample_weight=None)
    gradient_sw, hessian_sw = loss.gradient_hessian(y_true=y_true, raw_prediction=raw_prediction, sample_weight=sample_weight)
    if not loss.is_multiclass:
        assert_allclose(gradient * sample_weight, gradient_sw)
        assert_allclose(hessian * sample_weight, hessian_sw)
    else:
        assert_allclose(gradient * sample_weight[:, None], gradient_sw)
        assert_allclose(hessian * sample_weight[:, None], hessian_sw)