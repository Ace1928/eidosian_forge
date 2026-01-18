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
def test_vectorizer_min_df():
    test_data = ['abc', 'dea', 'eat']
    vect = CountVectorizer(analyzer='char', min_df=1)
    vect.fit(test_data)
    assert 'a' in vect.vocabulary_.keys()
    assert len(vect.vocabulary_.keys()) == 6
    assert len(vect.stop_words_) == 0
    vect.min_df = 2
    vect.fit(test_data)
    assert 'c' not in vect.vocabulary_.keys()
    assert len(vect.vocabulary_.keys()) == 2
    assert 'c' in vect.stop_words_
    assert len(vect.stop_words_) == 4
    vect.min_df = 0.8
    vect.fit(test_data)
    assert 'c' not in vect.vocabulary_.keys()
    assert len(vect.vocabulary_.keys()) == 1
    assert 'c' in vect.stop_words_
    assert len(vect.stop_words_) == 5