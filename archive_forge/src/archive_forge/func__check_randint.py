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
def _check_randint(self, func, ptr, max_width):
    """
        Check a randint()-like function.
        """
    ints = []
    for i in range(10):
        ints.append(func(5, 500000000))
    self.assertEqual(len(ints), len(set(ints)), ints)
    r = self._follow_cpython(ptr)
    for args in [(1, 5), (13, 5000), (20, 2 ** 62 + 2 ** 61)]:
        if args[1] > max_width:
            continue
        self._check_dist(func, r.randint, [args], niters=10)
    self.assertRaises(ValueError, func, 5, 4)
    self.assertRaises(ValueError, func, 5, 2)