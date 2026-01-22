from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage import data, filters, img_as_float
from skimage._shared.testing import run_in_parallel, expected_warnings
from skimage.segmentation import slic
def _check_segment_labels(seg1, seg2, allowed_mismatch_ratio=0.1):
    size = seg1.size
    ndiff = np.sum(seg1 != seg2)
    assert ndiff / size < allowed_mismatch_ratio