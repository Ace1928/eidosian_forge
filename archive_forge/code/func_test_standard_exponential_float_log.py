import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
def test_standard_exponential_float_log(self):
    randoms = self.rg.standard_exponential(10, dtype='float32', method='inv')
    assert_(len(randoms) == 10)
    assert randoms.dtype == np.float32
    params_0(partial(self.rg.standard_exponential, dtype='float32', method='inv'))