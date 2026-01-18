import itertools
import numpy as np
import pytest
from numpy.testing import (
from skimage._shared.testing import expected_warnings
from skimage.color.colorconv import hsv2rgb, rgb2hsv
from skimage.color.colorlabel import label2rgb
def test_wrong_kind():
    label = np.ones((3, 3))
    label2rgb(label, bg_label=-1)
    with pytest.raises(ValueError):
        label2rgb(label, kind='foo', bg_label=-1)