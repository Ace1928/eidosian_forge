import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
def test_reset_state_float(self):
    rg = Generator(self.bit_generator(*self.seed))
    rg.random(dtype='float32')
    state = rg.bit_generator.state
    n1 = rg.random(size=10, dtype='float32')
    rg2 = Generator(self.bit_generator())
    rg2.bit_generator.state = state
    n2 = rg2.random(size=10, dtype='float32')
    assert_((n1 == n2).all())