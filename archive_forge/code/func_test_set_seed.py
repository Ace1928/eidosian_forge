from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_allclose
import numpy as np
from skimage.data import camera
from skimage.util import random_noise, img_as_float
def test_set_seed():
    cam = camera()
    test = random_noise(cam, rng=42)
    assert_array_equal(test, random_noise(cam, rng=42))