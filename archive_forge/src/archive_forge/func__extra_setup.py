import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
@classmethod
def _extra_setup(cls):
    cls.vec_1d = np.arange(2.0, 102.0)
    cls.vec_2d = np.arange(2.0, 102.0)[None, :]
    cls.mat = np.arange(2.0, 102.0, 0.01).reshape((100, 100))
    cls.seed_error = TypeError