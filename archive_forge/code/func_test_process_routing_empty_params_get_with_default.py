import re
import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import (
from sklearn.linear_model import LinearRegression
from sklearn.tests.metadata_routing_common import (
from sklearn.utils import metadata_routing
from sklearn.utils._metadata_requests import (
from sklearn.utils.metadata_routing import (
from sklearn.utils.validation import check_is_fitted
@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('default', [None, 'default', []])
def test_process_routing_empty_params_get_with_default(method, default):
    empty_params = {}
    routed_params = process_routing(ConsumingClassifier(), 'fit', **empty_params)
    params_for_method = routed_params[method]
    assert isinstance(params_for_method, dict)
    assert set(params_for_method.keys()) == set(METHODS)
    default_params_for_method = routed_params.get(method, default=default)
    assert default_params_for_method == params_for_method