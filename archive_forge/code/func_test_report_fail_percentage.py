import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_report_fail_percentage(self):
    a = np.array([1, 1, 1, 1])
    b = np.array([1, 1, 1, 2])
    with pytest.raises(AssertionError) as exc_info:
        assert_allclose(a, b)
    msg = str(exc_info.value)
    assert_('Mismatched elements: 1 / 4 (25%)\nMax absolute difference: 1\nMax relative difference: 0.5' in msg)