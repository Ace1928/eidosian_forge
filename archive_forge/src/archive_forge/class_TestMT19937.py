import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
class TestMT19937(RNG):

    @classmethod
    def setup_class(cls):
        cls.bit_generator = MT19937
        cls.advance = None
        cls.seed = [2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rg = Generator(cls.bit_generator(*cls.seed))
        cls.initial_state = cls.rg.bit_generator.state
        cls.seed_vector_bits = 32
        cls._extra_setup()
        cls.seed_error = ValueError

    def test_numpy_state(self):
        nprg = np.random.RandomState()
        nprg.standard_normal(99)
        state = nprg.get_state()
        self.rg.bit_generator.state = state
        state2 = self.rg.bit_generator.state
        assert_((state[1] == state2['state']['key']).all())
        assert_(state[2] == state2['state']['pos'])