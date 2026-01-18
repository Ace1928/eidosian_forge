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
def test_reshape_zero_strides(self):
    a = np.ones(1)
    a = np.lib.stride_tricks.as_strided(a, shape=(5,), strides=(0,))
    assert_(a.reshape(5, 1).strides[0] == 0)