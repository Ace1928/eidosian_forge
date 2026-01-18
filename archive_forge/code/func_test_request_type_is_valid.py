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
@pytest.mark.parametrize('val, res', [(False, True), (True, True), (None, True), ('$UNUSED$', True), ('$WARN$', True), ('invalid-input', False), ('alias_arg', False)])
def test_request_type_is_valid(val, res):
    assert request_is_valid(val) == res