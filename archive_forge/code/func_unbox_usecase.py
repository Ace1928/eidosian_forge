import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def unbox_usecase(x):
    """
    Expect a set of numbers
    """
    res = 0
    for v in x:
        res += v
    return res