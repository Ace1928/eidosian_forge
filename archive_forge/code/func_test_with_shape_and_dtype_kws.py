import contextlib
import sys
import numpy as np
import random
import re
import threading
import gc
from numba.core.errors import TypingError
from numba import njit
from numba.core import types, utils, config
from numba.tests.support import MemoryLeakMixin, TestCase, tag, skip_if_32bit
import unittest
def test_with_shape_and_dtype_kws(self):
    for dtype in [np.int32, np.float32, np.complex64, np.dtype('complex64')]:
        self._test_with_shape_and_dtype_kw(dtype)