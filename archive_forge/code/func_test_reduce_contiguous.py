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
def test_reduce_contiguous(self):
    a = np.add.reduce(np.zeros((2, 1, 2)), (0, 1))
    b = np.add.reduce(np.zeros((2, 1, 2)), 1)
    assert_(a.flags.c_contiguous)
    assert_(a.flags.f_contiguous)
    assert_(b.flags.c_contiguous)