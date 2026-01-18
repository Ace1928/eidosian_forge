import os
from os.path import join
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_array_equal,
import pytest
from numpy.random import (
from numpy.random._common import interface
def uniform_from_uint64(x):
    return (x >> np.uint64(11)) * (1.0 / 9007199254740992.0)