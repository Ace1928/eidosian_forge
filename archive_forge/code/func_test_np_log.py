import warnings
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing._function_transformer import _get_adapter_from_container
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_np_log():
    X = np.arange(10).reshape((5, 2))
    assert_array_equal(FunctionTransformer(np.log1p).transform(X), np.log1p(X))