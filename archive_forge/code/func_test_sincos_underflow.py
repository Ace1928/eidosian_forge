import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
@pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
@pytest.mark.xfail(sys.platform.startswith('darwin'), reason="underflow is triggered for scalar 'sin'")
def test_sincos_underflow(self):
    with np.errstate(under='raise'):
        underflow_trigger = np.array(float.fromhex('0x1.f37f47a03f82ap-511'), dtype=np.float64)
        np.sin(underflow_trigger)
        np.cos(underflow_trigger)