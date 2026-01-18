import itertools
import numpy as np
import pytest
from numpy.testing import (
from skimage._shared.testing import expected_warnings
from skimage.color.colorconv import hsv2rgb, rgb2hsv
from skimage.color.colorlabel import label2rgb
def test_overlay_custom_saturation():
    rgb_img = np.random.uniform(size=(10, 10, 3))
    labels = np.ones((10, 10), dtype=np.int64)
    labels[5:, 5:] = 2
    labels[:3, :3] = 0
    alpha = 0.3
    saturation = 0.3
    rgb = label2rgb(labels, image=rgb_img, alpha=alpha, bg_label=0, saturation=saturation)
    hsv = rgb2hsv(rgb_img)
    hsv[..., 1] *= saturation
    saturaded_img = hsv2rgb(hsv)
    assert_array_almost_equal(saturaded_img[:3, :3] * (1 - alpha), rgb[:3, :3])