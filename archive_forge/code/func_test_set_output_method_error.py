import importlib
from collections import namedtuple
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn._config import config_context, get_config
from sklearn.preprocessing import StandardScaler
from sklearn.utils._set_output import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_set_output_method_error():
    """Check transform fails with invalid transform."""
    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    est = EstimatorWithSetOutput().fit(X)
    est.set_output(transform='bad')
    msg = 'output config must be in'
    with pytest.raises(ValueError, match=msg):
        est.transform(X)