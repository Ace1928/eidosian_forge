import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
def test_numpy_state(self):
    nprg = np.random.RandomState()
    nprg.standard_normal(99)
    state = nprg.get_state()
    self.rg.bit_generator.state = state
    state2 = self.rg.bit_generator.state
    assert_((state[1] == state2['state']['key']).all())
    assert_(state[2] == state2['state']['pos'])