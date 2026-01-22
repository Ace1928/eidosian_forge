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
def _check_array_results(self, func, pop, replace=True):
    """
        Check array results produced by *func* and their distribution.
        """
    n = len(pop)
    res = list(func().flat)
    self._check_results(pop, res, replace)
    dist = self._accumulate_array_results(func, n * 100)
    self._check_dist(pop, dist)