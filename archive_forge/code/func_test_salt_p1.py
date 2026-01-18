from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_allclose
import numpy as np
from skimage.data import camera
from skimage.util import random_noise, img_as_float
def test_salt_p1():
    image = np.random.rand(2, 3)
    noisy = random_noise(image, mode='salt', amount=1)
    assert_array_equal(noisy, [[1, 1, 1], [1, 1, 1]])