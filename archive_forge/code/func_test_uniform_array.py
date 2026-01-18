import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
def test_uniform_array(self):
    r = self.rg.uniform(np.array([-1.0] * 10), 0.0, size=10)
    assert_(len(r) == 10)
    assert_((r > -1).all())
    assert_((r <= 0).all())
    r = self.rg.uniform(np.array([-1.0] * 10), np.array([0.0] * 10), size=10)
    assert_(len(r) == 10)
    assert_((r > -1).all())
    assert_((r <= 0).all())
    r = self.rg.uniform(-1.0, np.array([0.0] * 10), size=10)
    assert_(len(r) == 10)
    assert_((r > -1).all())
    assert_((r <= 0).all())