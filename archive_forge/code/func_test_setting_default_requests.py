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
def test_setting_default_requests():
    test_cases = dict()

    class ExplicitRequest(BaseEstimator):
        __metadata_request__fit = {'prop': None}

        def fit(self, X, y, **kwargs):
            return self
    test_cases[ExplicitRequest] = {'prop': None}

    class ExplicitRequestOverwrite(BaseEstimator):
        __metadata_request__fit = {'prop': True}

        def fit(self, X, y, prop=None, **kwargs):
            return self
    test_cases[ExplicitRequestOverwrite] = {'prop': True}

    class ImplicitRequest(BaseEstimator):

        def fit(self, X, y, prop=None, **kwargs):
            return self
    test_cases[ImplicitRequest] = {'prop': None}

    class ImplicitRequestRemoval(BaseEstimator):
        __metadata_request__fit = {'prop': metadata_routing.UNUSED}

        def fit(self, X, y, prop=None, **kwargs):
            return self
    test_cases[ImplicitRequestRemoval] = {}
    for Klass, requests in test_cases.items():
        assert get_routing_for_object(Klass()).fit.requests == requests
        assert_request_is_empty(Klass().get_metadata_routing(), exclude='fit')
        Klass().fit(None, None)