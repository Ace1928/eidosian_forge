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
@pytest.mark.parametrize('factory', [CountVectorizer.build_analyzer, CountVectorizer.build_preprocessor, CountVectorizer.build_tokenizer])
def test_pickling_built_processors(factory):
    """Tokenizers cannot be pickled
    https://github.com/scikit-learn/scikit-learn/issues/12833
    """
    vec = CountVectorizer()
    function = factory(vec)
    text = "J'ai mangé du kangourou  ce midi, c'était pas très bon."
    roundtripped_function = pickle.loads(pickle.dumps(function))
    expected = function(text)
    result = roundtripped_function(text)
    assert result == expected