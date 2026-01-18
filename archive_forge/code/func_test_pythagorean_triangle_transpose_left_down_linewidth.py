import numpy as np
from ..._shared.testing import assert_equal, assert_almost_equal
from ..profile import profile_line
def test_pythagorean_triangle_transpose_left_down_linewidth():
    prof = profile_line(pyth_image.T[:, ::-1], (1, 4), (5, 1), linewidth=3, order=0, mode='constant')
    expected_prof = np.ones(6)
    assert_almost_equal(prof, expected_prof)