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
Test that when no metadata is passed, having a meta-estimator which does
    not yet support metadata routing works.

    Non-regression test for https://github.com/scikit-learn/scikit-learn/issues/28246
    