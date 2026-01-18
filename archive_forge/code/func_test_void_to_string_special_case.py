import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
def test_void_to_string_special_case(self):
    assert np.array([], dtype='V5').astype('S').dtype.itemsize == 5
    assert np.array([], dtype='V5').astype('U').dtype.itemsize == 4 * 5