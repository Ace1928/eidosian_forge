import pytest
import numpy as np
from skimage import data
from skimage.measure._label import _label_bool, label
from skimage.measure._ccomp import label_cython as clabel
from skimage._shared import testing
def test_return_num():
    img = data.binary_blobs(length=128, blob_size_fraction=0.15, n_dim=3)
    l_ndi = _label_bool(img, return_num=True)
    l_cy = clabel(img, return_num=True)
    testing.assert_equal(l_ndi, l_cy)