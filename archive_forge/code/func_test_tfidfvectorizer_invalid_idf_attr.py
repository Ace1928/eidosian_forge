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
def test_tfidfvectorizer_invalid_idf_attr():
    vect = TfidfVectorizer(use_idf=True)
    vect.fit(JUNK_FOOD_DOCS)
    copy = TfidfVectorizer(vocabulary=vect.vocabulary_, use_idf=True)
    expected_idf_len = len(vect.idf_)
    invalid_idf = [1.0] * (expected_idf_len + 1)
    with pytest.raises(ValueError):
        setattr(copy, 'idf_', invalid_idf)