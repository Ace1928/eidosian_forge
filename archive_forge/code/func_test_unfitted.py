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
def test_unfitted():
    X = 'foo!'
    clf = svm.SVC()
    with pytest.raises(Exception, match='.*\\bSVC\\b.*\\bnot\\b.*\\bfitted\\b'):
        clf.predict(X)
    clf = svm.NuSVR()
    with pytest.raises(Exception, match='.*\\bNuSVR\\b.*\\bnot\\b.*\\bfitted\\b'):
        clf.predict(X)