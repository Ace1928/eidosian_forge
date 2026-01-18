import html
import locale
import re
from contextlib import closing
from io import StringIO
from unittest.mock import patch
import pytest
from sklearn import config_context
from sklearn.base import BaseEstimator
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._estimator_html_repr import (
from sklearn.utils.fixes import parse_version
@pytest.mark.parametrize('final_estimator', [None, LinearSVC()])
def test_stacking_classifier(final_estimator):
    estimators = [('mlp', MLPClassifier(alpha=0.001)), ('tree', DecisionTreeClassifier())]
    clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
    html_output = estimator_html_repr(clf)
    assert html.escape(str(clf)) in html_output
    if final_estimator is None:
        assert 'LogisticRegression(' in html_output
    else:
        assert final_estimator.__class__.__name__ in html_output