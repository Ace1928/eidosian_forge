import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
def test_repr_roundtrip_foreign(self):
    o = 1.5
    assert_equal(o, np.longdouble(repr(o)))