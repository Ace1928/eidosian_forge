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
def test_subclass_int_tuple_assignment(self):

    class Subclass(np.ndarray):

        def __new__(cls, i):
            return np.ones((i,)).view(cls)
    x = Subclass(5)
    x[0,] = 2
    assert_equal(x[0], 2)