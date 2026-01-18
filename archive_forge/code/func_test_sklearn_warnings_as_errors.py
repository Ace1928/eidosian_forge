import atexit
import os
import unittest
import warnings
import numpy as np
import pytest
from scipy import sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import _IS_WASM
from sklearn.utils._testing import (
from sklearn.utils.deprecation import deprecated
from sklearn.utils.fixes import (
from sklearn.utils.metaestimators import available_if
@pytest.mark.parametrize('warning_info', _get_warnings_filters_info_list())
def test_sklearn_warnings_as_errors(warning_info):
    warnings_as_errors = os.environ.get('SKLEARN_WARNINGS_AS_ERRORS', '0') != '0'
    check_warnings_as_errors(warning_info, warnings_as_errors=warnings_as_errors)