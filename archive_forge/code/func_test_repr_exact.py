import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
def test_repr_exact():
    o = 1 + LD_INFO.eps
    assert_(repr(o) != '1')