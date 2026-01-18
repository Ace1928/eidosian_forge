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
def test_word_ngram_analyzer():
    cnga = CountVectorizer(analyzer='word', strip_accents='unicode', ngram_range=(3, 6)).build_analyzer()
    text = 'This \n\tis a test, really.\n\n I met Harry yesterday'
    expected = ['this is test', 'is test really', 'test really met']
    assert cnga(text)[:3] == expected
    expected = ['test really met harry yesterday', 'this is test really met harry', 'is test really met harry yesterday']
    assert cnga(text)[-3:] == expected
    cnga_file = CountVectorizer(input='file', analyzer='word', ngram_range=(3, 6)).build_analyzer()
    file = StringIO(text)
    assert cnga_file(file) == cnga(text)