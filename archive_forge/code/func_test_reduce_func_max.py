import numpy as np
from ..._shared.testing import assert_equal, assert_almost_equal
from ..profile import profile_line
def test_reduce_func_max():
    prof = profile_line(pyth_image, (0, 1), (3, 1), linewidth=3, order=0, reduce_func=np.max, mode='reflect')
    expected_prof = pyth_image[:4, :3].max(1)
    assert_almost_equal(prof, expected_prof)