import numpy as np
import pytest
from skimage import data
from skimage.restoration._rolling_ball import rolling_ball
from skimage.restoration._rolling_ball import ellipsoid_kernel
@pytest.mark.parametrize('radius', [2, 10, 12.5, 50])
def test_preserve_peaks(radius):
    x, y = np.meshgrid(range(100), range(100))
    img = 0 * x + 0 * y + 10
    img[10, 10] = 20
    img[20, 20] = 35
    img[45, 26] = 156
    expected_img = img - 10
    background = rolling_ball(img, radius=radius)
    assert np.allclose(img - background, expected_img)