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
def test_signed_integer_division_overflow(self):

    def test_type(t):
        min = np.array([np.iinfo(t).min])
        min //= -1
    with np.errstate(over='ignore'):
        for t in (np.int8, np.int16, np.int32, np.int64, int):
            test_type(t)