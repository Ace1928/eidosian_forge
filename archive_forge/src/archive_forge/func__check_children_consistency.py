import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
def _check_children_consistency(parent, left, right):
    assert parent.left_child is left
    assert parent.right_child is right
    assert len(left.sample_indices) + len(right.sample_indices) == len(parent.sample_indices)
    assert set(left.sample_indices).union(set(right.sample_indices)) == set(parent.sample_indices)
    assert set(left.sample_indices).intersection(set(right.sample_indices)) == set()