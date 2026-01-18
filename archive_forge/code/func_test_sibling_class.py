import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_sibling_class(self):
    for w1, w2 in itertools.product(self.wrappers, repeat=2):
        assert_(not np.issubdtype(w1(np.float32), w2(np.float64)))
        assert_(not np.issubdtype(w1(np.float64), w2(np.float32)))