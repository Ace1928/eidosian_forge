import numpy as np
from skimage.segmentation import join_segmentations, relabel_sequential
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
import pytest
def test_relabel_sequential_offset5_with0():
    ar = np.array([1, 1, 5, 5, 8, 99, 42, 0])
    ar_relab, fw, inv = relabel_sequential(ar, offset=5)
    _check_maps(ar, ar_relab, fw, inv)
    ar_relab_ref = np.array([5, 5, 6, 6, 7, 9, 8, 0])
    assert_array_equal(ar_relab, ar_relab_ref)
    fw_ref = np.zeros(100, int)
    fw_ref[1] = 5
    fw_ref[5] = 6
    fw_ref[8] = 7
    fw_ref[42] = 8
    fw_ref[99] = 9
    assert_array_equal(fw, fw_ref)
    inv_ref = np.array([0, 0, 0, 0, 0, 1, 5, 8, 42, 99])
    assert_array_equal(inv, inv_ref)