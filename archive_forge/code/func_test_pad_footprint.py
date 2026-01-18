import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import fetch
from skimage.morphology import footprints
@pytest.mark.parametrize('as_sequence', [tuple, None])
@pytest.mark.parametrize('pad_end', [True, False])
def test_pad_footprint(as_sequence, pad_end):
    footprint = np.array([[0, 0], [1, 0], [1, 1]], np.uint8)
    pad_width = [(0, 0), (0, 1)] if pad_end is True else [(0, 0), (1, 0)]
    expected_res = np.pad(footprint, pad_width)
    if as_sequence is not None:
        footprint = as_sequence([(footprint, 2), (footprint.T, 3)])
        expected_res = as_sequence([(expected_res, 2), (expected_res.T, 3)])
    actual_res = footprints.pad_footprint(footprint, pad_end=pad_end)
    assert type(expected_res) is type(actual_res)
    assert_equal(expected_res, actual_res)