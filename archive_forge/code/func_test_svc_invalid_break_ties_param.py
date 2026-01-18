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
def test_svc_invalid_break_ties_param(SVCClass):
    X, y = make_blobs(random_state=42)
    svm = SVCClass(kernel='linear', decision_function_shape='ovo', break_ties=True, random_state=42).fit(X, y)
    with pytest.raises(ValueError, match='break_ties must be False'):
        svm.predict(y)