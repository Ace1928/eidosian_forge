import re
import warnings
import numpy as np
import numpy.linalg as la
import pytest
from scipy import sparse, stats
from sklearn import datasets
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._data import BOUNDS_THRESHOLD, _handle_zeros_in_scale
from sklearn.svm import SVR
from sklearn.utils import gen_batches, shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import (
from sklearn.utils.sparsefuncs import mean_variance_axis
@pytest.mark.parametrize('array_namespace, device, dtype_name', yield_namespace_device_dtype_combinations())
@pytest.mark.parametrize('check', [check_array_api_input_and_values], ids=_get_check_estimator_ids)
@pytest.mark.parametrize('estimator', [MaxAbsScaler(), MinMaxScaler(), KernelCenterer(), Normalizer(norm='l1'), Normalizer(norm='l2'), Normalizer(norm='max')], ids=_get_check_estimator_ids)
def test_scaler_array_api_compliance(estimator, check, array_namespace, device, dtype_name):
    name = estimator.__class__.__name__
    check(name, estimator, array_namespace, device=device, dtype_name=dtype_name)