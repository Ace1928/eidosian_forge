import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
@pytest.mark.parametrize('arr', ([1.0, 2.0, 3.0], [1.0, np.nan, 3.0], np.nan, 0.0))
def test_subclass2(self, arr):
    """Check that we return subclasses, even if a NaN scalar."""

    class MySubclass(np.ndarray):
        pass
    m = np.median(np.array(arr).view(MySubclass))
    assert isinstance(m, MySubclass)