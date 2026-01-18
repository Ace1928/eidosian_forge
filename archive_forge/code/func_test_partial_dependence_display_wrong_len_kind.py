import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats.mstats import mquantiles
from sklearn.compose import make_column_transformer
from sklearn.datasets import (
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._testing import _convert_container
def test_partial_dependence_display_wrong_len_kind(pyplot, clf_diabetes, diabetes):
    """Check that we raise an error when `kind` is a list with a wrong length.

    This case can only be triggered using the `PartialDependenceDisplay.from_estimator`
    method.
    """
    disp = PartialDependenceDisplay.from_estimator(clf_diabetes, diabetes.data, features=[0, 2], grid_resolution=20, kind='average')
    disp.kind = ['average']
    err_msg = 'When `kind` is provided as a list of strings, it should contain as many elements as `features`. `kind` contains 1 element\\(s\\) and `features` contains 2 element\\(s\\).'
    with pytest.raises(ValueError, match=err_msg):
        disp.plot()