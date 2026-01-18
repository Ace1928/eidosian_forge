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
def test_packed_structure(self):

    class PackedStructure(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('a', ctypes.c_uint8), ('b', ctypes.c_uint16)]
    expected = np.dtype([('a', np.uint8), ('b', np.uint16)])
    self.check(PackedStructure, expected)