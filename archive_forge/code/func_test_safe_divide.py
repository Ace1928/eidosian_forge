import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn import datasets
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble._gb import _safe_divide
from sklearn.ensemble._gradient_boosting import predict_stages
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.svm import NuSVR
from sklearn.utils import check_random_state, tosequence
from sklearn.utils._mocking import NoSampleWeightWrapper
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
def test_safe_divide():
    """Test that _safe_divide handles division by zero."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert _safe_divide(np.float64(1e+300), 0) == 0
        assert _safe_divide(np.float64(0.0), np.float64(0.0)) == 0
    with pytest.warns(RuntimeWarning, match='overflow'):
        _safe_divide(np.float64(1e+300), 1e-10)