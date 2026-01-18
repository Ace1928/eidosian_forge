import re
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.optimize import check_grad
from sklearn import clone
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
def test_simple_example():
    """Test on a simple example.

    Puts four points in the input space where the opposite labels points are
    next to each other. After transform the samples from the same class
    should be next to each other.

    """
    X = np.array([[0, 0], [0, 1], [2, 0], [2, 1]])
    y = np.array([1, 0, 1, 0])
    nca = NeighborhoodComponentsAnalysis(n_components=2, init='identity', random_state=42)
    nca.fit(X, y)
    X_t = nca.transform(X)
    assert_array_equal(pairwise_distances(X_t).argsort()[:, 1], np.array([2, 3, 0, 1]))