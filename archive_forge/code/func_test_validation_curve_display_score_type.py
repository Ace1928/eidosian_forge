import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import (
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
@pytest.mark.parametrize('std_display_style', (None, 'errorbar'))
def test_validation_curve_display_score_type(pyplot, data, std_display_style):
    """Check the behaviour of setting the `score_type` parameter."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)
    param_name, param_range = ('max_depth', [1, 3, 5])
    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range)
    score_type = 'train'
    display = ValidationCurveDisplay.from_estimator(estimator, X, y, param_name=param_name, param_range=param_range, score_type=score_type, std_display_style=std_display_style)
    _, legend_label = display.ax_.get_legend_handles_labels()
    assert legend_label == ['Train']
    if std_display_style is None:
        assert len(display.lines_) == 1
        assert display.errorbar_ is None
        x_data, y_data = display.lines_[0].get_data()
    else:
        assert display.lines_ is None
        assert len(display.errorbar_) == 1
        x_data, y_data = display.errorbar_[0].lines[0].get_data()
    assert_array_equal(x_data, param_range)
    assert_allclose(y_data, train_scores.mean(axis=1))
    score_type = 'test'
    display = ValidationCurveDisplay.from_estimator(estimator, X, y, param_name=param_name, param_range=param_range, score_type=score_type, std_display_style=std_display_style)
    _, legend_label = display.ax_.get_legend_handles_labels()
    assert legend_label == ['Test']
    if std_display_style is None:
        assert len(display.lines_) == 1
        assert display.errorbar_ is None
        x_data, y_data = display.lines_[0].get_data()
    else:
        assert display.lines_ is None
        assert len(display.errorbar_) == 1
        x_data, y_data = display.errorbar_[0].lines[0].get_data()
    assert_array_equal(x_data, param_range)
    assert_allclose(y_data, test_scores.mean(axis=1))
    score_type = 'both'
    display = ValidationCurveDisplay.from_estimator(estimator, X, y, param_name=param_name, param_range=param_range, score_type=score_type, std_display_style=std_display_style)
    _, legend_label = display.ax_.get_legend_handles_labels()
    assert legend_label == ['Train', 'Test']
    if std_display_style is None:
        assert len(display.lines_) == 2
        assert display.errorbar_ is None
        x_data_train, y_data_train = display.lines_[0].get_data()
        x_data_test, y_data_test = display.lines_[1].get_data()
    else:
        assert display.lines_ is None
        assert len(display.errorbar_) == 2
        x_data_train, y_data_train = display.errorbar_[0].lines[0].get_data()
        x_data_test, y_data_test = display.errorbar_[1].lines[0].get_data()
    assert_array_equal(x_data_train, param_range)
    assert_allclose(y_data_train, train_scores.mean(axis=1))
    assert_array_equal(x_data_test, param_range)
    assert_allclose(y_data_test, test_scores.mean(axis=1))