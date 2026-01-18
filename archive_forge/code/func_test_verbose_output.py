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
def test_verbose_output():
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1, verbose=1, subsample=0.8)
    clf.fit(X, y)
    verbose_output = sys.stdout
    sys.stdout = old_stdout
    verbose_output.seek(0)
    header = verbose_output.readline().rstrip()
    true_header = ' '.join(['%10s'] + ['%16s'] * 3) % ('Iter', 'Train Loss', 'OOB Improve', 'Remaining Time')
    assert true_header == header
    n_lines = sum((1 for l in verbose_output.readlines()))
    assert 10 + 9 == n_lines