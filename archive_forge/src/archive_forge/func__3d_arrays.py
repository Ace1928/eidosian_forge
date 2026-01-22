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
def _3d_arrays(self):
    a = np.arange(24).reshape((4, 3, 2))
    b = a + 10
    c = (b + 10).copy(order='F')
    d = (c + 10)[::-1]
    e = (d + 10)[..., ::-1]
    return (a, b, c, d, e)