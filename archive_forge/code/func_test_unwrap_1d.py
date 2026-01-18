import numpy as np
from skimage.restoration import unwrap_phase
import sys
from skimage._shared import testing
from skimage._shared.testing import (
from skimage._shared._warnings import expected_warnings
def test_unwrap_1d():
    image = np.linspace(0, 10 * np.pi, 100)
    check_unwrap(image)
    with testing.raises(ValueError):
        check_unwrap(image, True)
    with testing.raises(ValueError):
        unwrap_phase(image, True, rng=0)