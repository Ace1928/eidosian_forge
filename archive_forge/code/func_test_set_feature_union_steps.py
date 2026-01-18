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
def test_set_feature_union_steps():
    mult2 = Mult(2)
    mult3 = Mult(3)
    mult5 = Mult(5)
    mult3.get_feature_names_out = lambda input_features: ['x3']
    mult2.get_feature_names_out = lambda input_features: ['x2']
    mult5.get_feature_names_out = lambda input_features: ['x5']
    ft = FeatureUnion([('m2', mult2), ('m3', mult3)])
    assert_array_equal([[2, 3]], ft.transform(np.asarray([[1]])))
    assert_array_equal(['m2__x2', 'm3__x3'], ft.get_feature_names_out())
    ft.transformer_list = [('m5', mult5)]
    assert_array_equal([[5]], ft.transform(np.asarray([[1]])))
    assert_array_equal(['m5__x5'], ft.get_feature_names_out())
    ft.set_params(transformer_list=[('mock', mult3)])
    assert_array_equal([[3]], ft.transform(np.asarray([[1]])))
    assert_array_equal(['mock__x3'], ft.get_feature_names_out())
    ft.set_params(mock=mult5)
    assert_array_equal([[5]], ft.transform(np.asarray([[1]])))
    assert_array_equal(['mock__x5'], ft.get_feature_names_out())