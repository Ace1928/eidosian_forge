import pickle
import re
import sys
from collections.abc import Iterable, Sized
from functools import partial
from io import StringIO
from itertools import chain, product
from types import GeneratorType
import numpy as np
import pytest
from scipy.stats import bernoulli, expon, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.cluster import KMeans
from sklearn.datasets import (
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import (
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import (
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection.tests.common import OneTimeSplitter
from sklearn.neighbors import KernelDensity, KNeighborsClassifier, LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tests.metadata_routing_common import (
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._mocking import CheckingClassifier, MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
def test_search_cv__pairwise_property_delegated_to_base_estimator():
    """
    Test implementation of BaseSearchCV has the pairwise property
    which matches the pairwise tag of its estimator.
    This test make sure pairwise tag is delegated to the base estimator.

    Non-regression test for issue #13920.
    """

    class EstimatorPairwise(BaseEstimator):

        def __init__(self, pairwise=True):
            self.pairwise = pairwise

        def _more_tags(self):
            return {'pairwise': self.pairwise}
    est = EstimatorPairwise()
    attr_message = 'BaseSearchCV _pairwise property must match estimator'
    for _pairwise_setting in [True, False]:
        est.set_params(pairwise=_pairwise_setting)
        cv = GridSearchCV(est, {'n_neighbors': [10]})
        assert _pairwise_setting == cv._get_tags()['pairwise'], attr_message