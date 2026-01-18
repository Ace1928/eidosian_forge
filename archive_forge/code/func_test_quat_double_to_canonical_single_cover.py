import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_quat_double_to_canonical_single_cover():
    x = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1], [-1, -1, -1, -1]])
    r = Rotation.from_quat(x)
    expected_quat = np.abs(x) / np.linalg.norm(x, axis=1)[:, None]
    assert_allclose(r.as_quat(canonical=True), expected_quat)