import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl

        Tests casts from and to string by checking the roundtripping property.

        The test also covers some string to string casts (but not all).

        If this test creates issues, it should possibly just be simplified
        or even removed (checking whether unaligned/non-contiguous casts give
        the same results is useful, though).
        