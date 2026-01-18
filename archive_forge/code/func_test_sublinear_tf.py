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
def test_sublinear_tf():
    X = [[1], [2], [3]]
    tr = TfidfTransformer(sublinear_tf=True, use_idf=False, norm=None)
    tfidf = tr.fit_transform(X).toarray()
    assert tfidf[0] == 1
    assert tfidf[1] > tfidf[0]
    assert tfidf[2] > tfidf[1]
    assert tfidf[1] < 2
    assert tfidf[2] < 3