import warnings
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets, linear_model
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._least_angle import _lars_path_residues
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import (
def test_singular_matrix():
    X1 = np.array([[1, 1.0], [1.0, 1.0]])
    y1 = np.array([1, 1])
    _, _, coef_path = linear_model.lars_path(X1, y1)
    assert_array_almost_equal(coef_path.T, [[0, 0], [1, 0]])