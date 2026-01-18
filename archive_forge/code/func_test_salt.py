from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_allclose
import numpy as np
from skimage.data import camera
from skimage.util import random_noise, img_as_float
def test_salt():
    cam = img_as_float(camera())
    amount = 0.15
    cam_noisy = random_noise(cam, rng=42, mode='salt', amount=amount)
    saltmask = cam != cam_noisy
    assert_allclose(cam_noisy[saltmask], np.ones(saltmask.sum()))
    proportion = float(saltmask.sum()) / (cam.shape[0] * cam.shape[1])
    tolerance = 0.01
    assert abs(amount - proportion) <= tolerance