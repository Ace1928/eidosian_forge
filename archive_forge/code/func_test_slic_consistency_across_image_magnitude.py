from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage import data, filters, img_as_float
from skimage._shared.testing import run_in_parallel, expected_warnings
from skimage.segmentation import slic
def test_slic_consistency_across_image_magnitude():
    img_uint8 = data.cat()[:256, :128]
    img_uint16 = 256 * img_uint8.astype(np.uint16)
    img_float32 = img_as_float(img_uint8)
    img_float32_norm = img_float32 / img_float32.max()
    img_float32_offset = img_float32 + 1000
    seg1 = slic(img_uint8)
    seg2 = slic(img_uint16)
    seg3 = slic(img_float32)
    seg4 = slic(img_float32_norm)
    seg5 = slic(img_float32_offset)
    np.testing.assert_array_equal(seg1, seg2)
    np.testing.assert_array_equal(seg1, seg3)
    np.testing.assert_array_equal(seg4, seg5)
    n_seg1 = seg1.max()
    n_seg4 = seg4.max()
    assert abs(n_seg1 - n_seg4) / n_seg1 < 0.5