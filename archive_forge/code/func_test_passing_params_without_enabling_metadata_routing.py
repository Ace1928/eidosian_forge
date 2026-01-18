import itertools
import os
import warnings
from functools import partial
import numpy as np
import pytest
from numpy.testing import (
from scipy import sparse
from sklearn import config_context
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.metrics import get_scorer, log_loss
from sklearn.model_selection import (
from sklearn.preprocessing import LabelEncoder, StandardScaler, scale
from sklearn.svm import l1_min_c
from sklearn.utils import _IS_32BIT, compute_class_weight, shuffle
from sklearn.utils._testing import ignore_warnings, skip_if_no_parallel
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
def test_passing_params_without_enabling_metadata_routing():
    """Test that the right error message is raised when metadata params
    are passed while not supported when `enable_metadata_routing=False`."""
    X, y = make_classification(n_samples=10, random_state=0)
    lr_cv = LogisticRegressionCV()
    msg = 'is only supported if enable_metadata_routing=True'
    with config_context(enable_metadata_routing=False):
        params = {'extra_param': 1.0}
        with pytest.raises(ValueError, match=msg):
            lr_cv.fit(X, y, **params)
        with pytest.raises(ValueError, match=msg):
            lr_cv.score(X, y, **params)