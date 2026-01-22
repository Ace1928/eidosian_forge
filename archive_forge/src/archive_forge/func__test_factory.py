import os
import numpy as np
from numpy.testing import suppress_warnings
import pytest
from scipy.special import (
from scipy.integrate import IntegrationWarning
from scipy.special._testutils import FuncData
def _test_factory(test, dtype=np.float64):
    """Boost test"""
    with suppress_warnings() as sup:
        sup.filter(IntegrationWarning, 'The occurrence of roundoff error is detected')
        with np.errstate(all='ignore'):
            test.check(dtype=dtype)