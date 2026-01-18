from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from sklearn._config import config_context
from sklearn.base import BaseEstimator
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
@pytest.mark.parametrize('array_namespace, device, dtype_name', yield_namespace_device_dtype_combinations())
@pytest.mark.parametrize('sample_weight, normalize, expected', [(None, False, 10.0), (None, True, 2.5), ([0.4, 0.4, 0.5, 0.7], False, 5.5), ([0.4, 0.4, 0.5, 0.7], True, 2.75), ([1, 2, 3, 4], False, 30.0), ([1, 2, 3, 4], True, 3.0)])
def test_weighted_sum(array_namespace, device, dtype_name, sample_weight, normalize, expected):
    xp = _array_api_for_tests(array_namespace, device)
    sample_score = numpy.asarray([1, 2, 3, 4], dtype=dtype_name)
    sample_score = xp.asarray(sample_score, device=device)
    if sample_weight is not None:
        sample_weight = numpy.asarray(sample_weight, dtype=dtype_name)
        sample_weight = xp.asarray(sample_weight, device=device)
    with config_context(array_api_dispatch=True):
        result = _weighted_sum(sample_score, sample_weight, normalize)
    assert isinstance(result, float)
    assert_allclose(result, expected, atol=_atol_for_type(dtype_name))