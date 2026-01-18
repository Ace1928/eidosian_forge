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
@pytest.mark.parametrize('SVCClass', [svm.SVC, svm.NuSVC])
def test_svc_ovr_tie_breaking(SVCClass):
    """Test if predict breaks ties in OVR mode.
    Related issue: https://github.com/scikit-learn/scikit-learn/issues/8277
    """
    X, y = make_blobs(random_state=0, n_samples=20, n_features=2)
    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    ys = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    xx, yy = np.meshgrid(xs, ys)
    common_params = dict(kernel='rbf', gamma=1000000.0, random_state=42, decision_function_shape='ovr')
    svm = SVCClass(break_ties=False, **common_params).fit(X, y)
    pred = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    dv = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    assert not np.all(pred == np.argmax(dv, axis=1))
    svm = SVCClass(break_ties=True, **common_params).fit(X, y)
    pred = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    dv = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    assert np.all(pred == np.argmax(dv, axis=1))