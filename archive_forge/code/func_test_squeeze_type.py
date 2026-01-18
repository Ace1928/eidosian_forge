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
def test_squeeze_type(self):
    a = np.array([3])
    b = np.array(3)
    assert_(type(a.squeeze()) is np.ndarray)
    assert_(type(b.squeeze()) is np.ndarray)