import itertools
import numpy as np
import pytest
from numpy.testing import (
from skimage._shared.testing import expected_warnings
from skimage.color.colorconv import hsv2rgb, rgb2hsv
from skimage.color.colorlabel import label2rgb
@pytest.mark.parametrize('channel_axis', [0, 1, -1])
def test_uint_image(channel_axis):
    img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    labels = np.zeros((10, 10), dtype=np.int64)
    labels[1:3, 1:3] = 1
    labels[6:9, 6:9] = 2
    output = label2rgb(labels, image=img, bg_label=0, channel_axis=channel_axis)
    assert np.issubdtype(output.dtype, np.floating)
    assert output.max() <= 1
    new_axis = channel_axis % output.ndim
    assert output.shape[new_axis] == 3