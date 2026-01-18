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
@pytest.mark.xfail(_IS_WASM, reason='no floating point exceptions, see https://github.com/numpy/numpy/pull/21895#issuecomment-1311525881')
def test_tfidf_no_smoothing():
    X = [[1, 1, 1], [1, 1, 0], [1, 0, 0]]
    tr = TfidfTransformer(smooth_idf=False, norm='l2')
    tfidf = tr.fit_transform(X).toarray()
    assert (tfidf >= 0).all()
    assert_array_almost_equal((tfidf ** 2).sum(axis=1), [1.0, 1.0, 1.0])
    X = [[1, 1, 0], [1, 1, 0], [1, 0, 0]]
    tr = TfidfTransformer(smooth_idf=False, norm='l2')
    in_warning_message = 'divide by zero'
    with pytest.warns(RuntimeWarning, match=in_warning_message):
        tr.fit_transform(X).toarray()