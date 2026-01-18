import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def test_prominence_condition(self):
    """
        Test prominence condition for peaks.
        """
    x = np.linspace(0, 10, 100)
    peaks_true = np.arange(1, 99, 2)
    offset = np.linspace(1, 10, peaks_true.size)
    x[peaks_true] += offset
    prominences = x[peaks_true] - x[peaks_true + 1]
    interval = (3, 9)
    keep = np.nonzero((interval[0] <= prominences) & (prominences <= interval[1]))
    peaks_calc, properties = find_peaks(x, prominence=interval)
    assert_equal(peaks_calc, peaks_true[keep])
    assert_equal(properties['prominences'], prominences[keep])
    assert_equal(properties['left_bases'], 0)
    assert_equal(properties['right_bases'], peaks_true[keep] + 1)