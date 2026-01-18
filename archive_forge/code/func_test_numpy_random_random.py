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
def test_numpy_random_random(self):
    cfunc = self._compile_array_dist('random', 1)
    size = (30, 30)
    res = cfunc(size)
    self.assertIsInstance(res, np.ndarray)
    self.assertEqual(res.shape, size)
    self.assertEqual(res.dtype, np.dtype('float64'))
    self.assertTrue(np.all(res >= 0.0))
    self.assertTrue(np.all(res < 1.0))
    self.assertTrue(np.any(res <= 0.1))
    self.assertTrue(np.any(res >= 0.9))
    mean = res.mean()
    self.assertGreaterEqual(mean, 0.45)
    self.assertLessEqual(mean, 0.55)