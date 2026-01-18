import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_randomstate_ctor_old_style_pickle():
    rs = np.random.RandomState(MT19937(0))
    rs.standard_normal(1)
    ctor, args, state_a = rs.__reduce__()
    assert args[:1] == ('MT19937',)
    b = ctor(*args[:1])
    b.set_state(state_a)
    state_b = b.get_state(legacy=False)
    assert_equal(state_a['bit_generator'], state_b['bit_generator'])
    assert_array_equal(state_a['state']['key'], state_b['state']['key'])
    assert_array_equal(state_a['state']['pos'], state_b['state']['pos'])
    assert_equal(state_a['has_gauss'], state_b['has_gauss'])
    assert_equal(state_a['gauss'], state_b['gauss'])