import sys
import operator
import pytest
import ctypes
import gc
import types
from typing import Any
import numpy as np
import numpy.dtypes
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import create_custom_field_dtype
from numpy.testing import (
from numpy.compat import pickle
from itertools import permutations
import random
import hypothesis
from hypothesis.extra import numpy as hynp
def test_subarray_cast_copies(self):
    arr = np.ones(3, dtype=[('f', 'i', 3)])
    cast = arr.astype(object)
    for fields in cast:
        assert type(fields) == tuple and len(fields) == 1
        subarr = fields[0]
        assert subarr.base is None
        assert subarr.flags.owndata