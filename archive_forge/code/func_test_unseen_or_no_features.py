from random import Random
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
def test_unseen_or_no_features():
    D = [{'camelot': 0, 'spamalot': 1}]
    for sparse in [True, False]:
        v = DictVectorizer(sparse=sparse).fit(D)
        X = v.transform({'push the pram a lot': 2})
        if sparse:
            X = X.toarray()
        assert_array_equal(X, np.zeros((1, 2)))
        X = v.transform({})
        if sparse:
            X = X.toarray()
        assert_array_equal(X, np.zeros((1, 2)))
        with pytest.raises(ValueError, match='empty'):
            v.transform([])