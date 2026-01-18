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
def test_zero_sized_array_indexing(self):
    tmp = np.array([])

    def index_tmp():
        tmp[np.array(10)]
    assert_raises(IndexError, index_tmp)