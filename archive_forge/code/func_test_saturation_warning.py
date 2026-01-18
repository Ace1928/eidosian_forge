import itertools
import numpy as np
import pytest
from numpy.testing import (
from skimage._shared.testing import expected_warnings
from skimage.color.colorconv import hsv2rgb, rgb2hsv
from skimage.color.colorlabel import label2rgb
def test_saturation_warning():
    rgb_img = np.random.uniform(size=(10, 10, 3))
    labels = np.ones((10, 10), dtype=np.int64)
    with expected_warnings(['saturation must be in range']):
        label2rgb(labels, image=rgb_img, bg_label=0, saturation=2)
    with expected_warnings(['saturation must be in range']):
        label2rgb(labels, image=rgb_img, bg_label=0, saturation=-1)