from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testAPI(self):
    assert_(not [m for m in dir(np.ndarray) if m not in dir(MaskedArray) and (not m.startswith('_'))])