import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
class Casting(enum.IntEnum):
    no = 0
    equiv = 1
    safe = 2
    same_kind = 3
    unsafe = 4