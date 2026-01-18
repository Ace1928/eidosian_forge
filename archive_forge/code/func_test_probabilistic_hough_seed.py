import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data, transform
from skimage._shared.testing import run_in_parallel
from skimage.draw import circle_perimeter, ellipse_perimeter, line
def test_probabilistic_hough_seed():
    image = data.checkerboard()
    lines = transform.probabilistic_hough_line(image, threshold=50, line_length=50, line_gap=1, rng=41537233)
    assert len(lines) == 56