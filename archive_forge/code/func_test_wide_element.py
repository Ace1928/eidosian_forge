import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_wide_element(self):
    a = np.array(['xxxxx'])
    assert_equal(np.array2string(a, max_line_width=5), "['xxxxx']")
    assert_equal(np.array2string(a, max_line_width=5, legacy='1.13'), "[ 'xxxxx']")