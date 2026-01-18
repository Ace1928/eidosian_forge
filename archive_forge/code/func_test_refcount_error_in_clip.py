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
def test_refcount_error_in_clip(self):
    a = np.zeros((2,), dtype='>i2').clip(min=0)
    x = a + a
    y = str(x)
    assert_(y == '[0 0]')