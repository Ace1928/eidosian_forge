import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import (
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
def test_validation_curve_display_default_usage(pyplot, data):
    """Check the default usage of the ValidationCurveDisplay class."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)
    param_name, param_range = ('max_depth', [1, 3, 5])
    display = ValidationCurveDisplay.from_estimator(estimator, X, y, param_name=param_name, param_range=param_range)
    import matplotlib as mpl
    assert display.errorbar_ is None
    assert isinstance(display.lines_, list)
    for line in display.lines_:
        assert isinstance(line, mpl.lines.Line2D)
    assert isinstance(display.fill_between_, list)
    for fill in display.fill_between_:
        assert isinstance(fill, mpl.collections.PolyCollection)
        assert fill.get_alpha() == 0.5
    assert display.score_name == 'Score'
    assert display.ax_.get_xlabel() == f'{param_name}'
    assert display.ax_.get_ylabel() == 'Score'
    _, legend_labels = display.ax_.get_legend_handles_labels()
    assert legend_labels == ['Train', 'Test']
    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range)
    assert_array_equal(display.param_range, param_range)
    assert_allclose(display.train_scores, train_scores)
    assert_allclose(display.test_scores, test_scores)