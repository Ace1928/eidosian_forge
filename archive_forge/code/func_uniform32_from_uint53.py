import os
from os.path import join
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_array_equal,
import pytest
from numpy.random import (
from numpy.random._common import interface
def uniform32_from_uint53(x):
    x = np.uint64(x) >> np.uint64(16)
    x = np.uint32(x & np.uint64(4294967295))
    return uint32_to_float32(x)