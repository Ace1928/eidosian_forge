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
def test_squeeze_contiguous(self):
    a = np.zeros((1, 2)).squeeze()
    b = np.zeros((2, 2, 2), order='F')[:, :, ::2].squeeze()
    assert_(a.flags.c_contiguous)
    assert_(a.flags.f_contiguous)
    assert_(b.flags.f_contiguous)