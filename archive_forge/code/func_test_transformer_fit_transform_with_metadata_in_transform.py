import pickle
import re
import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
import sklearn
from sklearn import config_context, datasets
from sklearn.base import (
from sklearn.decomposition import PCA
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._set_output import _get_output_config
from sklearn.utils._testing import (
@pytest.mark.usefixtures('enable_slep006')
def test_transformer_fit_transform_with_metadata_in_transform():
    """Test that having a transformer with metadata for transform raises a
    warning when calling fit_transform."""

    class CustomTransformer(BaseEstimator, TransformerMixin):

        def fit(self, X, y=None, prop=None):
            return self

        def transform(self, X, prop=None):
            return X
    with pytest.warns(UserWarning, match='`transform` method which consumes metadata'):
        CustomTransformer().set_transform_request(prop=True).fit_transform([[1]], [1], prop=1)
    with warnings.catch_warnings(record=True) as record:
        CustomTransformer().set_transform_request(prop=True).fit_transform([[1]], [1])
        assert len(record) == 0