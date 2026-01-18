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
def test_ufunc_no_unnecessary_views(self):

    class Subclass(np.ndarray):
        pass
    x = np.array([1, 2, 3]).view(Subclass)
    y = np.add(x, x, x)
    assert_equal(id(x), id(y))