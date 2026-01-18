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
@pytest.mark.parametrize('passthrough', [None, 'passthrough'])
def test_set_pipeline_step_passthrough(passthrough):
    X = np.array([[1]])
    y = np.array([1])
    mult2 = Mult(mult=2)
    mult3 = Mult(mult=3)
    mult5 = Mult(mult=5)

    def make():
        return Pipeline([('m2', mult2), ('m3', mult3), ('last', mult5)])
    pipeline = make()
    exp = 2 * 3 * 5
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    assert_array_equal([exp], pipeline.fit(X).predict(X))
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))
    pipeline.set_params(m3=passthrough)
    exp = 2 * 5
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    assert_array_equal([exp], pipeline.fit(X).predict(X))
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))
    assert pipeline.get_params(deep=True) == {'steps': pipeline.steps, 'm2': mult2, 'm3': passthrough, 'last': mult5, 'memory': None, 'm2__mult': 2, 'last__mult': 5, 'verbose': False}
    pipeline.set_params(m2=passthrough)
    exp = 5
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    assert_array_equal([exp], pipeline.fit(X).predict(X))
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))
    other_methods = ['predict_proba', 'predict_log_proba', 'decision_function', 'transform', 'score']
    for method in other_methods:
        getattr(pipeline, method)(X)
    pipeline.set_params(m2=mult2)
    exp = 2 * 5
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    assert_array_equal([exp], pipeline.fit(X).predict(X))
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))
    pipeline = make()
    pipeline.set_params(last=passthrough)
    exp = 6
    assert_array_equal([[exp]], pipeline.fit(X, y).transform(X))
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))
    inner_msg = "'str' object has no attribute 'predict'"
    outer_msg = "This 'Pipeline' has no attribute 'predict'"
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        getattr(pipeline, 'predict')
    assert isinstance(exec_info.value.__cause__, AttributeError)
    assert inner_msg in str(exec_info.value.__cause__)
    exp = 2 * 5
    pipeline = Pipeline([('m2', mult2), ('m3', passthrough), ('last', mult5)])
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    assert_array_equal([exp], pipeline.fit(X).predict(X))
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))