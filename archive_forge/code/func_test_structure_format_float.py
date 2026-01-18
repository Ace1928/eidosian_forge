import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_structure_format_float(self):
    array_scalar = np.array((1.0, 2.1234567890123457, 3.0), dtype='f8,f8,f8')
    assert_equal(np.array2string(array_scalar), '(1., 2.12345679, 3.)')