import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import (
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
@pytest.mark.parametrize('Display, params', [(LearningCurveDisplay, {}), (ValidationCurveDisplay, {'param_name': 'max_depth', 'param_range': [1, 3, 5]})])
def test_subclassing_displays(pyplot, data, Display, params):
    """Check that named constructors return the correct type when subclassed.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/pull/27675
    """
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)

    class SubclassOfDisplay(Display):
        pass
    display = SubclassOfDisplay.from_estimator(estimator, X, y, **params)
    assert isinstance(display, SubclassOfDisplay)