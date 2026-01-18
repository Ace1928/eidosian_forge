import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle
def test_numpy_float_python_long_addition(self):
    a = np.float_(23.0) + 2 ** 135
    assert_equal(a, 23.0 + 2 ** 135)