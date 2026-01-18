import numpy as np
from skimage.restoration import unwrap_phase
import sys
from skimage._shared import testing
from skimage._shared.testing import (
from skimage._shared._warnings import expected_warnings
def test_unwrap_3d_middle_wrap_around():
    image = np.zeros((20, 30, 40), dtype=np.float32)
    unwrap = unwrap_phase(image, wrap_around=[False, True, False])
    assert_(np.all(unwrap == 0))