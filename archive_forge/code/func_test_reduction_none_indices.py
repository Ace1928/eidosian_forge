import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_reduction_none_indices():
    result = Rotation.identity().reduce(return_indices=True)
    assert type(result) == tuple
    assert len(result) == 3
    reduced, left_best, right_best = result
    assert left_best is None
    assert right_best is None