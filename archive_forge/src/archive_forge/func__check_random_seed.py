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
def _check_random_seed(self, seedfunc, randomfunc):
    """
        Check seed()- and random()-like functions.
        """
    r = np.random.RandomState()
    for i in [0, 1, 125, 2 ** 32 - 1]:
        r.seed(np.uint32(i))
        seedfunc(i)
        for j in range(N + 10):
            self.assertPreciseEqual(randomfunc(), r.uniform(0.0, 1.0))