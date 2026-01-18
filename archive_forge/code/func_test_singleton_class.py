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
def test_singleton_class():
    X = iris_data
    y = iris_target
    singleton_class = 1
    ind_singleton, = np.where(y == singleton_class)
    y[ind_singleton] = 2
    y[ind_singleton[0]] = singleton_class
    nca = NeighborhoodComponentsAnalysis(max_iter=30)
    nca.fit(X, y)
    ind_1, = np.where(y == 1)
    ind_2, = np.where(y == 2)
    y[ind_1] = 0
    y[ind_1[0]] = 1
    y[ind_2] = 0
    y[ind_2[0]] = 2
    nca = NeighborhoodComponentsAnalysis(max_iter=30)
    nca.fit(X, y)
    ind_0, = np.where(y == 0)
    ind_1, = np.where(y == 1)
    ind_2, = np.where(y == 2)
    X = X[[ind_0[0], ind_1[0], ind_2[0]]]
    y = y[[ind_0[0], ind_1[0], ind_2[0]]]
    nca = NeighborhoodComponentsAnalysis(init='identity', max_iter=30)
    nca.fit(X, y)
    assert_array_equal(X, nca.transform(X))