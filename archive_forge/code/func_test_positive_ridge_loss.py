import warnings
from itertools import product
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets
from sklearn.datasets import (
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._ridge import (
from sklearn.metrics import get_scorer, make_scorer, mean_squared_error
from sklearn.model_selection import (
from sklearn.preprocessing import minmax_scale
from sklearn.utils import _IS_32BIT, check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('alpha', [0.001, 0.01, 0.1, 1.0])
def test_positive_ridge_loss(alpha):
    """Check ridge loss consistency when positive argument is enabled."""
    X, y = make_regression(n_samples=300, n_features=300, random_state=42)
    alpha = 0.1
    n_checks = 100

    def ridge_loss(model, random_state=None, noise_scale=1e-08):
        intercept = model.intercept_
        if random_state is not None:
            rng = np.random.RandomState(random_state)
            coef = model.coef_ + rng.uniform(0, noise_scale, size=model.coef_.shape)
        else:
            coef = model.coef_
        return 0.5 * np.sum((y - X @ coef - intercept) ** 2) + 0.5 * alpha * np.sum(coef ** 2)
    model = Ridge(alpha=alpha).fit(X, y)
    model_positive = Ridge(alpha=alpha, positive=True).fit(X, y)
    loss = ridge_loss(model)
    loss_positive = ridge_loss(model_positive)
    assert loss <= loss_positive
    for random_state in range(n_checks):
        loss_perturbed = ridge_loss(model_positive, random_state=random_state)
        assert loss_positive <= loss_perturbed