import itertools
import re
import shutil
import time
from tempfile import mkdtemp
import joblib
import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_classifier
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tests.metadata_routing_common import (
from sklearn.utils._metadata_requests import COMPOSITE_METHODS, METHODS
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import check_is_fitted
def test_pipeline_get_feature_names_out_passes_names_through():
    """Check that pipeline passes names through.

    Non-regresion test for #21349.
    """
    X, y = (iris.data, iris.target)

    class AddPrefixStandardScalar(StandardScaler):

        def get_feature_names_out(self, input_features=None):
            names = super().get_feature_names_out(input_features=input_features)
            return np.asarray([f'my_prefix_{name}' for name in names], dtype=object)
    pipe = make_pipeline(AddPrefixStandardScalar(), StandardScaler())
    pipe.fit(X, y)
    input_names = iris.feature_names
    feature_names_out = pipe.get_feature_names_out(input_names)
    assert_array_equal(feature_names_out, [f'my_prefix_{name}' for name in input_names])