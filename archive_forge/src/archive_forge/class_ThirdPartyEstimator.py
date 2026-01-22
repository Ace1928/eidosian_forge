from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
class ThirdPartyEstimator(_Estimator):

    def __init__(self, b):
        self.b = b
        super().__init__(a=0)

    def fit(self, X=None, y=None):
        super().fit(X, y)