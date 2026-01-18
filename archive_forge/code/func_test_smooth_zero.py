import re
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
def test_smooth_zero():
    """Check edge case with zero smoothing and cv does not contain category."""
    X = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]).T
    y = np.array([2.1, 4.3, 1.2, 3.1, 1.0, 9.0, 10.3, 14.2, 13.3, 15.0])
    enc = TargetEncoder(smooth=0.0, shuffle=False, cv=2)
    X_trans = enc.fit_transform(X, y)
    assert_allclose(X_trans[0], np.mean(y[5:]))
    assert_allclose(X_trans[-1], np.mean(y[:5]))