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
def test_word_analyzer_unigrams_and_bigrams():
    wa = CountVectorizer(analyzer='word', strip_accents='unicode', ngram_range=(1, 2)).build_analyzer()
    text = "J'ai mangé du kangourou  ce midi, c'était pas très bon."
    expected = ['ai', 'mange', 'du', 'kangourou', 'ce', 'midi', 'etait', 'pas', 'tres', 'bon', 'ai mange', 'mange du', 'du kangourou', 'kangourou ce', 'ce midi', 'midi etait', 'etait pas', 'pas tres', 'tres bon']
    assert wa(text) == expected