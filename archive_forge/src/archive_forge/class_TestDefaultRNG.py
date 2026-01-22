import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
class TestDefaultRNG(RNG):

    @classmethod
    def setup_class(cls):
        cls.bit_generator = PCG64
        cls.advance = 2 ** 63 + 2 ** 31 + 2 ** 15 + 1
        cls.seed = [12345]
        cls.rg = np.random.default_rng(*cls.seed)
        cls.initial_state = cls.rg.bit_generator.state
        cls.seed_vector_bits = 64
        cls._extra_setup()

    def test_default_is_pcg64(self):
        assert_(isinstance(self.rg.bit_generator, PCG64))

    def test_seed(self):
        np.random.default_rng()
        np.random.default_rng(None)
        np.random.default_rng(12345)
        np.random.default_rng(0)
        np.random.default_rng(43660444402423911716352051725018508569)
        np.random.default_rng([43660444402423911716352051725018508569, 279705150948142787361475340226491943209])
        with pytest.raises(ValueError):
            np.random.default_rng(-1)
        with pytest.raises(ValueError):
            np.random.default_rng([12345, -1])