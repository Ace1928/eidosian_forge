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
def test_transformation_dimensions():
    X = np.arange(12).reshape(4, 3)
    y = [1, 1, 2, 2]
    transformation = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        NeighborhoodComponentsAnalysis(init=transformation).fit(X, y)
    transformation = np.array([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(ValueError):
        NeighborhoodComponentsAnalysis(init=transformation).fit(X, y)
    transformation = np.arange(9).reshape(3, 3)
    NeighborhoodComponentsAnalysis(init=transformation).fit(X, y)