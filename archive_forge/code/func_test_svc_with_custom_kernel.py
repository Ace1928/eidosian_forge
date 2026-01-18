import numpy as np
import pytest
from scipy import sparse
from sklearn import base, datasets, linear_model, svm
from sklearn.datasets import load_digits, make_blobs, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm.tests import test_svm
from sklearn.utils._testing import (
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.fixes import (
@pytest.mark.parametrize('lil_container', LIL_CONTAINERS)
def test_svc_with_custom_kernel(lil_container):

    def kfunc(x, y):
        return safe_sparse_dot(x, y.T)
    X_sp = lil_container(X)
    clf_lin = svm.SVC(kernel='linear').fit(X_sp, Y)
    clf_mylin = svm.SVC(kernel=kfunc).fit(X_sp, Y)
    assert_array_equal(clf_lin.predict(X_sp), clf_mylin.predict(X_sp))