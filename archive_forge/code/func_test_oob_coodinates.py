import numpy as np
from ..._shared.testing import assert_equal, assert_almost_equal
from ..profile import profile_line
def test_oob_coodinates():
    offset = 2
    idx = pyth_image.shape[0] + offset
    prof = profile_line(pyth_image, (-offset, 2), (idx, 2), linewidth=1, order=0, reduce_func=None, mode='constant')
    expected_prof = np.vstack([np.zeros((offset, 1)), pyth_image[:, 2, np.newaxis], np.zeros((offset + 1, 1))])
    assert_almost_equal(prof, expected_prof)