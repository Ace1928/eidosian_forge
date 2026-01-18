import os
from os.path import join
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_array_equal,
import pytest
from numpy.random import (
from numpy.random._common import interface
def test_seed_float_array(self):
    assert_raises(TypeError, self.bit_generator, np.array([np.pi]))
    assert_raises(TypeError, self.bit_generator, np.array([-np.pi]))
    assert_raises(TypeError, self.bit_generator, np.array([np.pi, -np.pi]))
    assert_raises(TypeError, self.bit_generator, np.array([0, np.pi]))
    assert_raises(TypeError, self.bit_generator, [np.pi])
    assert_raises(TypeError, self.bit_generator, [0, np.pi])