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
@fails_if_pypy
def test_vectorizer_stop_words_inconsistent():
    lstr = "\\['and', 'll', 've'\\]"
    message = 'Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens %s not in stop_words.' % lstr
    for vec in [CountVectorizer(), TfidfVectorizer(), HashingVectorizer()]:
        vec.set_params(stop_words=["you've", 'you', "you'll", 'AND'])
        with pytest.warns(UserWarning, match=message):
            vec.fit_transform(['hello world'])
        del vec._stop_words_id
        assert _check_stop_words_consistency(vec) is False
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        vec.fit_transform(['hello world'])
    assert _check_stop_words_consistency(vec) is None
    vec.set_params(stop_words=["you've", 'you', "you'll", 'blah', 'AND'])
    with pytest.warns(UserWarning, match=message):
        vec.fit_transform(['hello world'])