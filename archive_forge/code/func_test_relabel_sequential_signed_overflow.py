import numpy as np
from skimage.segmentation import join_segmentations, relabel_sequential
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
import pytest
def test_relabel_sequential_signed_overflow():
    imax = np.iinfo(np.int32).max
    labels = np.array([0, 1, 99, 42, 42], dtype=np.int32)
    output, fw, inv = relabel_sequential(labels, offset=imax)
    reference = np.array([0, imax, imax + 2, imax + 1, imax + 1], dtype=np.uint32)
    assert_array_equal(output, reference)
    assert output.dtype == reference.dtype