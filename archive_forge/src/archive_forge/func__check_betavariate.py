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
def _check_betavariate(self, func, ptr):
    """
        Check a betavariate()-like function.
        """
    r = self._follow_cpython(ptr)
    self._check_dist(func, r.betavariate, [(0.5, 2.5)])
    self.assertRaises(ValueError, func, 0.0, 1.0)
    self.assertRaises(ValueError, func, 1.0, 0.0)
    self.assertRaises(ValueError, func, -0.5, 1.0)
    self.assertRaises(ValueError, func, 1.0, -0.5)