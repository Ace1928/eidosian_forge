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
def test_search_sorted_invalid_arguments(self):
    x = np.arange(0, 4, dtype='datetime64[D]')
    assert_raises(TypeError, x.searchsorted, 1)