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
def test_removing_non_existing_param_raises():
    """Test that removing a metadata using UNUSED which doesn't exist raises."""

    class InvalidRequestRemoval(BaseEstimator):
        __metadata_request__fit = {'prop': metadata_routing.UNUSED}

        def fit(self, X, y, **kwargs):
            return self
    with pytest.raises(ValueError, match='Trying to remove parameter'):
        InvalidRequestRemoval().get_metadata_routing()