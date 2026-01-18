import os
from os.path import join
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_array_equal,
import pytest
from numpy.random import (
from numpy.random._common import interface
def test_set_key(self):
    bit_generator = self.bit_generator(*self.data1['seed'])
    state = bit_generator.state
    keyed = self.bit_generator(counter=state['state']['counter'], key=state['state']['key'])
    assert_state_equal(bit_generator.state, keyed.state)