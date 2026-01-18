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
def test_structured_count_nonzero(self):
    arr = np.array([0, 1]).astype('i4, (2)i4')[:1]
    count = np.count_nonzero(arr)
    assert_equal(count, 0)