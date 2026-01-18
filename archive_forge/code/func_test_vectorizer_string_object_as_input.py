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
@pytest.mark.parametrize('Vectorizer', (CountVectorizer, TfidfVectorizer, HashingVectorizer))
def test_vectorizer_string_object_as_input(Vectorizer):
    message = 'Iterable over raw text documents expected, string object received.'
    vec = Vectorizer()
    with pytest.raises(ValueError, match=message):
        vec.fit_transform('hello world!')
    with pytest.raises(ValueError, match=message):
        vec.fit('hello world!')
    vec.fit(['some text', 'some other text'])
    with pytest.raises(ValueError, match=message):
        vec.transform('hello world!')