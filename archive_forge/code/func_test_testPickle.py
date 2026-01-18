from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testPickle(self):
    x = arange(12)
    x[4:10:2] = masked
    x = x.reshape(4, 3)
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        s = pickle.dumps(x, protocol=proto)
        y = pickle.loads(s)
        assert_(eq(x, y))