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
def _check_choice_1(self, a, pop):
    """
        Check choice(a) against pop.
        """
    cfunc = jit(nopython=True)(numpy_choice1)
    n = len(pop)
    res = [cfunc(a) for i in range(n)]
    self._check_results(pop, res)
    dist = [cfunc(a) for i in range(n * 100)]
    self._check_dist(pop, dist)