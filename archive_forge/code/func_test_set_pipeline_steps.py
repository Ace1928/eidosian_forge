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
def test_set_pipeline_steps():
    transf1 = Transf()
    transf2 = Transf()
    pipeline = Pipeline([('mock', transf1)])
    assert pipeline.named_steps['mock'] is transf1
    pipeline.steps = [('mock2', transf2)]
    assert 'mock' not in pipeline.named_steps
    assert pipeline.named_steps['mock2'] is transf2
    assert [('mock2', transf2)] == pipeline.steps
    pipeline.set_params(steps=[('mock', transf1)])
    assert [('mock', transf1)] == pipeline.steps
    pipeline.set_params(mock=transf2)
    assert [('mock', transf2)] == pipeline.steps
    pipeline.set_params(steps=[('junk', ())])
    msg = re.escape("Last step of Pipeline should implement fit or be the string 'passthrough'.")
    with pytest.raises(TypeError, match=msg):
        pipeline.fit([[1]], [1])
    msg = "This 'Pipeline' has no attribute 'fit_transform'"
    with pytest.raises(AttributeError, match=msg):
        pipeline.fit_transform([[1]], [1])