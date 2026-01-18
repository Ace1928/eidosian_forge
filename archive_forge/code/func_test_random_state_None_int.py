import numpy as np
from numpy.testing import (assert_allclose,
import pytest
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import distance
from scipy._lib._util import check_random_state
@pytest.mark.parametrize('seed', [None, 27870671])
def test_random_state_None_int(self, seed):
    rs = check_random_state(None)
    old_global_state = rs.get_state()
    directed_hausdorff(self.path_1, self.path_2, seed)
    rs2 = check_random_state(None)
    new_global_state = rs2.get_state()
    assert_equal(new_global_state, old_global_state)