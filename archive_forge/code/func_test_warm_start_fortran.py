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
@pytest.mark.parametrize('Cls', GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_fortran(Cls, global_random_seed):
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=global_random_seed)
    est_c = Cls(n_estimators=1, random_state=global_random_seed, warm_start=True)
    est_fortran = Cls(n_estimators=1, random_state=global_random_seed, warm_start=True)
    est_c.fit(X, y)
    est_c.set_params(n_estimators=11)
    est_c.fit(X, y)
    X_fortran = np.asfortranarray(X)
    est_fortran.fit(X_fortran, y)
    est_fortran.set_params(n_estimators=11)
    est_fortran.fit(X_fortran, y)
    assert_allclose(est_c.predict(X), est_fortran.predict(X))