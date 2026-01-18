import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_array_almost_equal
from scipy.special import comb
from sklearn.utils._random import _our_rand_r_py
from sklearn.utils.random import _random_choice_csc, sample_without_replacement
def test_random_choice_csc(n_samples=10000, random_state=24):
    classes = [np.array([0, 1]), np.array([0, 1, 2])]
    class_probabilities = [np.array([0.5, 0.5]), np.array([0.6, 0.1, 0.3])]
    got = _random_choice_csc(n_samples, classes, class_probabilities, random_state)
    assert sp.issparse(got)
    for k in range(len(classes)):
        p = np.bincount(got.getcol(k).toarray().ravel()) / float(n_samples)
        assert_array_almost_equal(class_probabilities[k], p, decimal=1)
    classes = [[0, 1], [1, 2]]
    class_probabilities = [np.array([0.5, 0.5]), np.array([0, 1 / 2, 1 / 2])]
    got = _random_choice_csc(n_samples=n_samples, classes=classes, random_state=random_state)
    assert sp.issparse(got)
    for k in range(len(classes)):
        p = np.bincount(got.getcol(k).toarray().ravel()) / float(n_samples)
        assert_array_almost_equal(class_probabilities[k], p, decimal=1)
    classes = [np.array([0, 1]), np.array([0, 1, 2])]
    class_probabilities = [np.array([0.0, 1.0]), np.array([0.0, 1.0, 0.0])]
    got = _random_choice_csc(n_samples, classes, class_probabilities, random_state)
    assert sp.issparse(got)
    for k in range(len(classes)):
        p = np.bincount(got.getcol(k).toarray().ravel(), minlength=len(class_probabilities[k])) / n_samples
        assert_array_almost_equal(class_probabilities[k], p, decimal=1)
    classes = [[1], [0]]
    class_probabilities = [np.array([0.0, 1.0]), np.array([1.0])]
    got = _random_choice_csc(n_samples=n_samples, classes=classes, random_state=random_state)
    assert sp.issparse(got)
    for k in range(len(classes)):
        p = np.bincount(got.getcol(k).toarray().ravel()) / n_samples
        assert_array_almost_equal(class_probabilities[k], p, decimal=1)