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
def _check_getrandbits(self, func, ptr):
    """
        Check a getrandbits()-like function.
        """
    r = self._follow_cpython(ptr)
    for nbits in range(1, 65):
        expected = r.getrandbits(nbits)
        got = func(nbits)
        self.assertPreciseEqual(expected, got)
    self.assertRaises(OverflowError, func, 65)
    self.assertRaises(OverflowError, func, 9999999)
    self.assertRaises(OverflowError, func, -1)