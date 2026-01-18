import os
from os.path import join
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_array_equal,
import pytest
from numpy.random import (
from numpy.random._common import interface
def test_seedsequence():
    from numpy.random.bit_generator import ISeedSequence, ISpawnableSeedSequence, SeedlessSeedSequence
    s1 = SeedSequence(range(10), spawn_key=(1, 2), pool_size=6)
    s1.spawn(10)
    s2 = SeedSequence(**s1.state)
    assert_equal(s1.state, s2.state)
    assert_equal(s1.n_children_spawned, s2.n_children_spawned)
    assert_raises(TypeError, ISeedSequence)
    assert_raises(TypeError, ISpawnableSeedSequence)
    dummy = SeedlessSeedSequence()
    assert_raises(NotImplementedError, dummy.generate_state, 10)
    assert len(dummy.spawn(10)) == 10