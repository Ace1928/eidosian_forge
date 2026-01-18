import re
import numpy as np
import pytest
from numpy.testing import (
from sklearn import base, datasets, linear_model, metrics, svm
from sklearn.datasets import make_blobs, make_classification
from sklearn.exceptions import (
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import (  # type: ignore
from sklearn.svm._classes import _validate_dual_parameter
from sklearn.utils import check_random_state, shuffle
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import CSR_CONTAINERS, LIL_CONTAINERS
from sklearn.utils.validation import _num_samples
def test_svr_coef_sign():
    X = np.random.RandomState(21).randn(10, 3)
    y = np.random.RandomState(12).randn(10)
    for svr in [svm.SVR(kernel='linear'), svm.NuSVR(kernel='linear'), svm.LinearSVR(dual='auto')]:
        svr.fit(X, y)
        assert_array_almost_equal(svr.predict(X), np.dot(X, svr.coef_.ravel()) + svr.intercept_)