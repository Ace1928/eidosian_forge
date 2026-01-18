import numpy as np
from skimage.segmentation import join_segmentations, relabel_sequential
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
import pytest
def test_relabel_sequential_int_dtype_overflow():
    ar = np.array([1, 3, 0, 2, 5, 4], dtype=np.uint8)
    offset = 254
    ar_relab, fw, inv = relabel_sequential(ar, offset=offset)
    _check_maps(ar, ar_relab, fw, inv)
    assert all((a.dtype == np.uint16 for a in (ar_relab, fw)))
    assert inv.dtype == ar.dtype
    ar_relab_ref = np.where(ar > 0, ar.astype(int) + offset - 1, 0)
    assert_array_equal(ar_relab, ar_relab_ref)