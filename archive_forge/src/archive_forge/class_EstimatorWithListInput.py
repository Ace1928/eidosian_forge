import importlib
from collections import namedtuple
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn._config import config_context, get_config
from sklearn.preprocessing import StandardScaler
from sklearn.utils._set_output import (
from sklearn.utils.fixes import CSR_CONTAINERS
class EstimatorWithListInput(_SetOutputMixin):

    def fit(self, X, y=None):
        assert isinstance(X, list)
        self.n_features_in_ = len(X[0])
        return self

    def transform(self, X, y=None):
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f'X{i}' for i in range(self.n_features_in_)], dtype=object)