import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data, transform
from skimage._shared.testing import run_in_parallel
from skimage.draw import circle_perimeter, ellipse_perimeter, line
def test_probabilistic_hough():
    img = np.zeros((100, 100), dtype=int)
    for i in range(25, 75):
        img[100 - i, i] = 100
        img[i, i] = 100
    theta = np.linspace(0, np.pi, 45)
    lines = transform.probabilistic_hough_line(img, threshold=10, line_length=10, line_gap=1, theta=theta)
    sorted_lines = []
    for ln in lines:
        ln = list(ln)
        ln.sort(key=lambda x: x[0])
        sorted_lines.append(ln)
    assert [(25, 75), (74, 26)] in sorted_lines
    assert [(25, 25), (74, 74)] in sorted_lines
    transform.probabilistic_hough_line(img, line_length=10, line_gap=3)