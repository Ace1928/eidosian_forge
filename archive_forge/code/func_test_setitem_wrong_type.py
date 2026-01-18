import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_setitem_wrong_type():
    r = Rotation.random(10, random_state=0)
    with pytest.raises(TypeError, match='Rotation object'):
        r[0] = 1