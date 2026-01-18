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
def test_none_metadata_passed():
    """Test that passing None as metadata when not requested doesn't raise"""
    MetaRegressor(estimator=ConsumingRegressor()).fit(X, y, sample_weight=None)