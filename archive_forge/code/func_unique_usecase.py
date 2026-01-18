import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def unique_usecase(src):
    seen = set()
    res = []
    for v in src:
        if v not in seen:
            seen.add(v)
            res.append(v)
    return res