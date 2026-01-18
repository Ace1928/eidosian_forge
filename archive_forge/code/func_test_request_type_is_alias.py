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
@pytest.mark.parametrize('val, res', [(False, False), (True, False), (None, False), ('$UNUSED$', False), ('$WARN$', False), ('invalid-input', False), ('valid_arg', True)])
def test_request_type_is_alias(val, res):
    assert request_is_alias(val) == res