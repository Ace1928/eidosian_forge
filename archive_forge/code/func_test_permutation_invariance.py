import copy
import pickle
import warnings
import numpy as np
import pytest
from scipy.special import expit
import sklearn
from sklearn.datasets import make_regression
from sklearn.isotonic import (
from sklearn.utils import shuffle
from sklearn.utils._testing import (
from sklearn.utils.validation import check_array
def test_permutation_invariance():
    ir = IsotonicRegression()
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [1, 41, 51, 1, 2, 5, 24]
    sample_weight = [1, 2, 3, 4, 5, 6, 7]
    x_s, y_s, sample_weight_s = shuffle(x, y, sample_weight, random_state=0)
    y_transformed = ir.fit_transform(x, y, sample_weight=sample_weight)
    y_transformed_s = ir.fit(x_s, y_s, sample_weight=sample_weight_s).transform(x)
    assert_array_equal(y_transformed, y_transformed_s)