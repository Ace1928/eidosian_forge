import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
def test_output_filling_uniform(self):
    rg = self.rg
    state = rg.bit_generator.state
    size = (31, 7, 97)
    existing = np.empty(size)
    rg.bit_generator.state = state
    rg.random(out=existing)
    rg.bit_generator.state = state
    direct = rg.random(size=size)
    assert_equal(direct, existing)
    existing = np.empty(size, dtype=np.float32)
    rg.bit_generator.state = state
    rg.random(out=existing, dtype=np.float32)
    rg.bit_generator.state = state
    direct = rg.random(size=size, dtype=np.float32)
    assert_equal(direct, existing)