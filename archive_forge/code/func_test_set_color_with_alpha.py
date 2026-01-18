import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
from skimage._shared.testing import run_in_parallel
from skimage._shared._dependency_checks import has_mpl
from skimage.draw import (
from skimage.measure import regionprops
def test_set_color_with_alpha():
    img = np.zeros((10, 10))
    rr, cc, alpha = line_aa(0, 0, 0, 30)
    set_color(img, (rr, cc), 1, alpha=alpha)
    with pytest.raises(ValueError):
        set_color(img, (rr, cc), (255, 0, 0), alpha=alpha)
    img = np.zeros((10, 10, 3))
    rr, cc, alpha = line_aa(0, 0, 0, 30)
    set_color(img, (rr, cc), (1, 0, 0), alpha=alpha)