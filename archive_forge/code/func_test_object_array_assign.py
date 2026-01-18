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
def test_object_array_assign(self):
    x = np.empty((2, 2), object)
    x.flat[2] = (1, 2, 3)
    assert_equal(x.flat[2], (1, 2, 3))