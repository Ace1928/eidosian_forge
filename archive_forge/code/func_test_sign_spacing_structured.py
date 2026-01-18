import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_sign_spacing_structured(self):
    a = np.ones(2, dtype='<f,<f')
    assert_equal(repr(a), "array([(1., 1.), (1., 1.)], dtype=[('f0', '<f4'), ('f1', '<f4')])")
    assert_equal(repr(a[0]), '(1., 1.)')