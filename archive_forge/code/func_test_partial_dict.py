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
def test_partial_dict(self):
    assert_raises(ValueError, np.dtype, {'formats': ['i4', 'i4'], 'f0': ('i4', 0), 'f1': ('i4', 4)})