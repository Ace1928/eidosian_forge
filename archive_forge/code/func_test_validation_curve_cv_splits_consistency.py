import os
import re
import sys
import tempfile
import warnings
from functools import partial
from io import StringIO
from time import sleep
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.datasets import (
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import (
from sklearn.model_selection import (
from sklearn.model_selection._validation import (
from sklearn.model_selection.tests.common import OneTimeSplitter
from sklearn.model_selection.tests.test_search import FailingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.svm import SVC, LinearSVC
from sklearn.tests.metadata_routing_common import (
from sklearn.utils import shuffle
from sklearn.utils._mocking import CheckingClassifier, MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
def test_validation_curve_cv_splits_consistency():
    n_samples = 100
    n_splits = 5
    X, y = make_classification(n_samples=100, random_state=0)
    scores1 = validation_curve(SVC(kernel='linear', random_state=0), X, y, param_name='C', param_range=[0.1, 0.1, 0.2, 0.2], cv=OneTimeSplitter(n_splits=n_splits, n_samples=n_samples))
    assert_array_almost_equal(*np.vsplit(np.hstack(scores1)[(0, 2, 1, 3), :], 2))
    scores2 = validation_curve(SVC(kernel='linear', random_state=0), X, y, param_name='C', param_range=[0.1, 0.1, 0.2, 0.2], cv=KFold(n_splits=n_splits, shuffle=True))
    assert_array_almost_equal(*np.vsplit(np.hstack(scores2)[(0, 2, 1, 3), :], 2))
    scores3 = validation_curve(SVC(kernel='linear', random_state=0), X, y, param_name='C', param_range=[0.1, 0.1, 0.2, 0.2], cv=KFold(n_splits=n_splits))
    assert_array_almost_equal(np.array(scores3), np.array(scores1))