import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_seed_alt_bit_gen(restore_singleton_bitgen):
    bg = PCG64(0)
    np.random.set_bit_generator(bg)
    state = np.random.get_state(legacy=False)
    np.random.seed(1)
    new_state = np.random.get_state(legacy=False)
    print(state)
    print(new_state)
    assert state['bit_generator'] == 'PCG64'
    assert state['state']['state'] != new_state['state']['state']
    assert state['state']['inc'] != new_state['state']['inc']