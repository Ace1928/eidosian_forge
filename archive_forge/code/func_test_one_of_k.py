from random import Random
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
def test_one_of_k():
    D_in = [{'version': '1', 'ham': 2}, {'version': '2', 'spam': 0.3}, {'version=3': True, 'spam': -1}]
    v = DictVectorizer()
    X = v.fit_transform(D_in)
    assert X.shape == (3, 5)
    D_out = v.inverse_transform(X)
    assert D_out[0] == {'version=1': 1, 'ham': 2}
    names = v.get_feature_names_out()
    assert 'version=2' in names
    assert 'version' not in names