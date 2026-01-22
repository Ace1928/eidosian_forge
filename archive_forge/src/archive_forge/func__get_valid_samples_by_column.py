import warnings
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import (
def _get_valid_samples_by_column(X, col):
    """Get non NaN samples in column of X"""
    return X[:, [col]][~np.isnan(X[:, col])]