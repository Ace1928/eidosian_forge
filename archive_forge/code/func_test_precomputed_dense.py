import re
import warnings
from itertools import product
import joblib
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import (
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning, NotFittedError
from sklearn.metrics._dist_metrics import (
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS, pairwise_distances
from sklearn.metrics.tests.test_dist_metrics import BOOL_METRICS
from sklearn.metrics.tests.test_pairwise_distances_reduction import (
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import (
from sklearn.neighbors._base import (
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.validation import check_random_state
def test_precomputed_dense():

    def make_train_test(X_train, X_test):
        return (metrics.pairwise_distances(X_train), metrics.pairwise_distances(X_test, X_train))
    estimators = [neighbors.KNeighborsClassifier, neighbors.KNeighborsRegressor, neighbors.RadiusNeighborsClassifier, neighbors.RadiusNeighborsRegressor]
    check_precomputed(make_train_test, estimators)