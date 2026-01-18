import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_subclass_backwards(self):
    for w in self.wrappers:
        assert_(not np.issubdtype(np.floating, w(np.float32)))
        assert_(not np.issubdtype(np.floating, w(np.float64)))