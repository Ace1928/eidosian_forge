import collections
import functools
import math
import multiprocessing
import os
import random
import subprocess
import sys
import threading
import itertools
from textwrap import dedent
import numpy as np
import unittest
import numba
from numba import jit, _helperlib, njit
from numba.core import types
from numba.tests.support import TestCase, compile_function, tag
from numba.core.errors import TypingError
def _check_randrange(self, func1, func2, func3, ptr, max_width, is_numpy, tp=None):
    """
        Check a randrange()-like function.
        """
    ints = []
    for i in range(10):
        ints.append(func1(500000000))
        ints.append(func2(5, 500000000))
        if func3 is not None:
            ints.append(func3(5, 500000000, 3))
    if is_numpy:
        rr = self._follow_numpy(ptr).randint
    else:
        rr = self._follow_cpython(ptr).randrange
    widths = [w for w in [1, 5, 8, 5000, 2 ** 40, 2 ** 62 + 2 ** 61] if w < max_width]
    pydtype = tp if is_numpy else None
    for width in widths:
        self._check_dist(func1, rr, [(width,)], niters=10, pydtype=pydtype)
        self._check_dist(func2, rr, [(-2, 2 + width)], niters=10, pydtype=pydtype)
        if func3 is not None:
            self.assertPreciseEqual(func3(-2, 2 + width, 6), rr(-2, 2 + width, 6))
            self.assertPreciseEqual(func3(2 + width, 2, -3), rr(2 + width, 2, -3))
    self.assertRaises(ValueError, func1, 0)
    self.assertRaises(ValueError, func1, -5)
    self.assertRaises(ValueError, func2, 5, 5)
    self.assertRaises(ValueError, func2, 5, 2)
    if func3 is not None:
        self.assertRaises(ValueError, func3, 5, 7, -1)
        self.assertRaises(ValueError, func3, 7, 5, 1)