import sys
from contextlib import nullcontext
import numpy as np
import pytest
from packaging.version import Version
from ..casting import (
from ..testing import suppress_warnings
def test_usable_binary128():
    yes = have_binary128()
    with np.errstate(over='ignore'):
        exp_test = np.longdouble(2) ** 16383
    assert yes == (exp_test.dtype.itemsize == 16 and np.isfinite(exp_test) and _check_nmant(np.longdouble, 112))