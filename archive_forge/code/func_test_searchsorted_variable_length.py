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
def test_searchsorted_variable_length(self):
    x = np.array(['a', 'aa', 'b'])
    y = np.array(['d', 'e'])
    assert_equal(x.searchsorted(y), [3, 3])