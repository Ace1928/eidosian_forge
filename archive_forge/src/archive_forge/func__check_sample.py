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
def _check_sample(self, size, sample):
    if size is not None:
        self.assertIsInstance(sample, np.ndarray)
        self.assertEqual(sample.dtype, np.float64)
        if isinstance(size, int):
            self.assertEqual(sample.shape, (size,))
        else:
            self.assertEqual(sample.shape, size)
    else:
        self.assertIsInstance(sample, float)
    for val in np.nditer(sample):
        self.assertGreaterEqual(val, 0)