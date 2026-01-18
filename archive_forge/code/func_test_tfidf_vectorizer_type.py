import pickle
import re
import warnings
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from io import StringIO
from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import sparse
from sklearn.base import clone
from sklearn.feature_extraction.text import (
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils import _IS_WASM, IS_PYPY
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('vectorizer_dtype, output_dtype, warning_expected', [(np.int32, np.float64, True), (np.int64, np.float64, True), (np.float32, np.float32, False), (np.float64, np.float64, False)])
def test_tfidf_vectorizer_type(vectorizer_dtype, output_dtype, warning_expected):
    X = np.array(['numpy', 'scipy', 'sklearn'])
    vectorizer = TfidfVectorizer(dtype=vectorizer_dtype)
    warning_msg_match = "'dtype' should be used."
    if warning_expected:
        with pytest.warns(UserWarning, match=warning_msg_match):
            X_idf = vectorizer.fit_transform(X)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            X_idf = vectorizer.fit_transform(X)
    assert X_idf.dtype == output_dtype