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
def test_repeat_broadcasting(self):
    a = np.arange(60).reshape(3, 4, 5)
    for axis in chain(range(-a.ndim, a.ndim), [None]):
        assert_equal(a.repeat(2, axis=axis), a.repeat([2], axis=axis))