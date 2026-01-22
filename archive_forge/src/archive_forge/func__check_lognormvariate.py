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
def _check_lognormvariate(self, func2, func1, func0, ptr):
    """
        Check a lognormvariate()-like function.
        """
    r = self._follow_numpy(ptr)
    if func2 is not None:
        self._check_dist(func2, r.lognormal, [(1.0, 1.0), (2.0, 0.5), (-2.0, 0.5)], niters=N // 2 + 10)
    if func1 is not None:
        self._check_dist(func1, r.lognormal, [(0.5,)])
    if func0 is not None:
        self._check_dist(func0, r.lognormal, [()])