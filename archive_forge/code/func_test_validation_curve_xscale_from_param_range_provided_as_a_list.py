import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import (
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
@pytest.mark.parametrize('param_range, xscale', [([5, 10, 15], 'linear'), ([-50, 5, 50, 500], 'symlog'), ([5, 50, 500], 'log')])
def test_validation_curve_xscale_from_param_range_provided_as_a_list(pyplot, data, param_range, xscale):
    """Check the induced xscale from the provided param_range values."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)
    param_name = 'max_depth'
    display = ValidationCurveDisplay.from_estimator(estimator, X, y, param_name=param_name, param_range=param_range)
    assert display.ax_.get_xscale() == xscale