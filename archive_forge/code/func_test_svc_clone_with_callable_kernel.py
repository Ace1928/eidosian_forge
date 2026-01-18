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
def test_svc_clone_with_callable_kernel():
    svm_callable = svm.SVC(kernel=lambda x, y: np.dot(x, y.T), probability=True, random_state=0, decision_function_shape='ovr')
    svm_cloned = base.clone(svm_callable)
    svm_cloned.fit(iris.data, iris.target)
    svm_builtin = svm.SVC(kernel='linear', probability=True, random_state=0, decision_function_shape='ovr')
    svm_builtin.fit(iris.data, iris.target)
    assert_array_almost_equal(svm_cloned.dual_coef_, svm_builtin.dual_coef_)
    assert_array_almost_equal(svm_cloned.intercept_, svm_builtin.intercept_)
    assert_array_equal(svm_cloned.predict(iris.data), svm_builtin.predict(iris.data))
    assert_array_almost_equal(svm_cloned.predict_proba(iris.data), svm_builtin.predict_proba(iris.data), decimal=4)
    assert_array_almost_equal(svm_cloned.decision_function(iris.data), svm_builtin.decision_function(iris.data))