import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def update_usecase(a, b, c):
    s = set()
    s.update(a)
    s.update(b)
    s.update(c)
    return list(s)