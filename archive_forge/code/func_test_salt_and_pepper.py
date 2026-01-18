from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_allclose
import numpy as np
from skimage.data import camera
from skimage.util import random_noise, img_as_float
def test_salt_and_pepper():
    cam = img_as_float(camera())
    amount = 0.15
    cam_noisy = random_noise(cam, rng=42, mode='s&p', amount=amount, salt_vs_pepper=0.25)
    saltmask = np.logical_and(cam != cam_noisy, cam_noisy == 1.0)
    peppermask = np.logical_and(cam != cam_noisy, cam_noisy == 0.0)
    assert_allclose(cam_noisy[saltmask], np.ones(saltmask.sum()))
    assert_allclose(cam_noisy[peppermask], np.zeros(peppermask.sum()))
    proportion = float(saltmask.sum() + peppermask.sum()) / (cam.shape[0] * cam.shape[1])
    tolerance = 0.01
    assert abs(amount - proportion) <= tolerance
    assert 0.18 < saltmask.sum() / peppermask.sum() < 0.35