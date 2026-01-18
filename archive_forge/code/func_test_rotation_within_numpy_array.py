import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_rotation_within_numpy_array():
    single = Rotation.random(random_state=0)
    multiple = Rotation.random(2, random_state=1)
    array = np.array(single)
    assert_equal(array.shape, ())
    array = np.array(multiple)
    assert_equal(array.shape, (2,))
    assert_allclose(array[0].as_matrix(), multiple[0].as_matrix())
    assert_allclose(array[1].as_matrix(), multiple[1].as_matrix())
    array = np.array([single])
    assert_equal(array.shape, (1,))
    assert_equal(array[0], single)
    array = np.array([multiple])
    assert_equal(array.shape, (1, 2))
    assert_allclose(array[0, 0].as_matrix(), multiple[0].as_matrix())
    assert_allclose(array[0, 1].as_matrix(), multiple[1].as_matrix())
    array = np.array([single, multiple], dtype=object)
    assert_equal(array.shape, (2,))
    assert_equal(array[0], single)
    assert_equal(array[1], multiple)
    array = np.array([multiple, multiple, multiple])
    assert_equal(array.shape, (3, 2))