import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def test_threshold_condition(self):
    """
        Test threshold condition for peaks.
        """
    x = (0, 2, 1, 4, -1)
    peaks, props = find_peaks(x, threshold=(None, None))
    assert_equal(peaks, np.array([1, 3]))
    assert_equal(props['left_thresholds'], np.array([2, 3]))
    assert_equal(props['right_thresholds'], np.array([1, 5]))
    assert_equal(find_peaks(x, threshold=2)[0], np.array([3]))
    assert_equal(find_peaks(x, threshold=3.5)[0], np.array([]))
    assert_equal(find_peaks(x, threshold=(None, 5))[0], np.array([1, 3]))
    assert_equal(find_peaks(x, threshold=(None, 4))[0], np.array([1]))
    assert_equal(find_peaks(x, threshold=(2, 4))[0], np.array([]))