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
@pytest.mark.parametrize('metric', [r2_score, d2_tweedie_score, d2_pinball_score])
def test_regression_single_sample(metric):
    y_true = [0]
    y_pred = [1]
    warning_msg = 'not well-defined with less than two samples.'
    with pytest.warns(UndefinedMetricWarning, match=warning_msg):
        score = metric(y_true, y_pred)
        assert np.isnan(score)