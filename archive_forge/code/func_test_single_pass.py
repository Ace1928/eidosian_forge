import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def test_single_pass(self):
    distances = [0, 1, 2, 5]
    gaps = [0, 1, 2, 0, 1]
    test_matr = np.zeros([20, 50]) + 1e-12
    length = 12
    line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
    test_matr[line[0], line[1]] = 1
    max_distances = np.full(20, max(distances))
    identified_lines = _identify_ridge_lines(test_matr, max_distances, max(gaps) + 1)
    assert_array_equal(identified_lines, [line])