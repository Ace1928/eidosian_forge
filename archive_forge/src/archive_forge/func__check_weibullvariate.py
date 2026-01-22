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
def _check_weibullvariate(self, func2, func1, ptr):
    """
        Check a weibullvariate()-like function.
        """
    r = self._follow_cpython(ptr)
    if func2 is not None:
        self._check_dist(func2, r.weibullvariate, [(0.5, 2.5)])
    if func1 is not None:
        for i in range(3):
            self.assertPreciseEqual(func1(2.5), r.weibullvariate(1.0, 2.5))