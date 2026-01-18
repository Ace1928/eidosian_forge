from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import optimize
from scipy.special import factorial, xlogy
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
from sklearn.metrics._regression import _check_reg_targets
from sklearn.model_selection import GridSearchCV
from sklearn.utils._testing import (
def test_pinball_loss_relation_with_mae():
    rng = np.random.RandomState(714)
    n = 100
    y_true = rng.normal(size=n)
    y_pred = y_true.copy() + rng.uniform(n)
    assert mean_absolute_error(y_true, y_pred) == mean_pinball_loss(y_true, y_pred, alpha=0.5) * 2