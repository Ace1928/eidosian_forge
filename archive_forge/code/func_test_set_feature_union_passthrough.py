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
def test_set_feature_union_passthrough():
    """Check the behaviour of setting a transformer to `"passthrough"`."""
    mult2 = Mult(2)
    mult3 = Mult(3)
    mult2.get_feature_names_out = lambda input_features: ['x2']
    mult3.get_feature_names_out = lambda input_features: ['x3']
    X = np.asarray([[1]])
    ft = FeatureUnion([('m2', mult2), ('m3', mult3)])
    assert_array_equal([[2, 3]], ft.fit(X).transform(X))
    assert_array_equal([[2, 3]], ft.fit_transform(X))
    assert_array_equal(['m2__x2', 'm3__x3'], ft.get_feature_names_out())
    ft.set_params(m2='passthrough')
    assert_array_equal([[1, 3]], ft.fit(X).transform(X))
    assert_array_equal([[1, 3]], ft.fit_transform(X))
    assert_array_equal(['m2__myfeat', 'm3__x3'], ft.get_feature_names_out(['myfeat']))
    ft.set_params(m3='passthrough')
    assert_array_equal([[1, 1]], ft.fit(X).transform(X))
    assert_array_equal([[1, 1]], ft.fit_transform(X))
    assert_array_equal(['m2__myfeat', 'm3__myfeat'], ft.get_feature_names_out(['myfeat']))
    ft.set_params(m3=mult3)
    assert_array_equal([[1, 3]], ft.fit(X).transform(X))
    assert_array_equal([[1, 3]], ft.fit_transform(X))
    assert_array_equal(['m2__myfeat', 'm3__x3'], ft.get_feature_names_out(['myfeat']))
    ft = FeatureUnion([('m2', 'passthrough'), ('m3', mult3)])
    assert_array_equal([[1, 3]], ft.fit(X).transform(X))
    assert_array_equal([[1, 3]], ft.fit_transform(X))
    assert_array_equal(['m2__myfeat', 'm3__x3'], ft.get_feature_names_out(['myfeat']))
    X = iris.data
    columns = X.shape[1]
    pca = PCA(n_components=2, svd_solver='randomized', random_state=0)
    ft = FeatureUnion([('passthrough', 'passthrough'), ('pca', pca)])
    assert_array_equal(X, ft.fit(X).transform(X)[:, :columns])
    assert_array_equal(X, ft.fit_transform(X)[:, :columns])
    assert_array_equal(['passthrough__f0', 'passthrough__f1', 'passthrough__f2', 'passthrough__f3', 'pca__pca0', 'pca__pca1'], ft.get_feature_names_out(['f0', 'f1', 'f2', 'f3']))
    ft.set_params(pca='passthrough')
    X_ft = ft.fit(X).transform(X)
    assert_array_equal(X_ft, np.hstack([X, X]))
    X_ft = ft.fit_transform(X)
    assert_array_equal(X_ft, np.hstack([X, X]))
    assert_array_equal(['passthrough__f0', 'passthrough__f1', 'passthrough__f2', 'passthrough__f3', 'pca__f0', 'pca__f1', 'pca__f2', 'pca__f3'], ft.get_feature_names_out(['f0', 'f1', 'f2', 'f3']))
    ft.set_params(passthrough=pca)
    assert_array_equal(X, ft.fit(X).transform(X)[:, -columns:])
    assert_array_equal(X, ft.fit_transform(X)[:, -columns:])
    assert_array_equal(['passthrough__pca0', 'passthrough__pca1', 'pca__f0', 'pca__f1', 'pca__f2', 'pca__f3'], ft.get_feature_names_out(['f0', 'f1', 'f2', 'f3']))
    ft = FeatureUnion([('passthrough', 'passthrough'), ('pca', pca)], transformer_weights={'passthrough': 2})
    assert_array_equal(X * 2, ft.fit(X).transform(X)[:, :columns])
    assert_array_equal(X * 2, ft.fit_transform(X)[:, :columns])
    assert_array_equal(['passthrough__f0', 'passthrough__f1', 'passthrough__f2', 'passthrough__f3', 'pca__pca0', 'pca__pca1'], ft.get_feature_names_out(['f0', 'f1', 'f2', 'f3']))