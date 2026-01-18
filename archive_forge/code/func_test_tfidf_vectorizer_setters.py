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
def test_tfidf_vectorizer_setters():
    norm, use_idf, smooth_idf, sublinear_tf = ('l2', False, False, False)
    tv = TfidfVectorizer(norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
    tv.fit(JUNK_FOOD_DOCS)
    assert tv._tfidf.norm == norm
    assert tv._tfidf.use_idf == use_idf
    assert tv._tfidf.smooth_idf == smooth_idf
    assert tv._tfidf.sublinear_tf == sublinear_tf
    tv.norm = 'l1'
    tv.use_idf = True
    tv.smooth_idf = True
    tv.sublinear_tf = True
    assert tv._tfidf.norm == norm
    assert tv._tfidf.use_idf == use_idf
    assert tv._tfidf.smooth_idf == smooth_idf
    assert tv._tfidf.sublinear_tf == sublinear_tf
    tv.fit(JUNK_FOOD_DOCS)
    assert tv._tfidf.norm == tv.norm
    assert tv._tfidf.use_idf == tv.use_idf
    assert tv._tfidf.smooth_idf == tv.smooth_idf
    assert tv._tfidf.sublinear_tf == tv.sublinear_tf