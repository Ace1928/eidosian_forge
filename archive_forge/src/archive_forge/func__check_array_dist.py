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
def _check_array_dist(self, funcname, scalar_args):
    """
        Check returning an array according to a given distribution.
        """
    cfunc = self._compile_array_dist(funcname, len(scalar_args) + 1)
    r = self._follow_numpy(get_np_state_ptr())
    pyfunc = getattr(r, funcname)
    for size in (8, (2, 3)):
        args = scalar_args + (size,)
        expected = pyfunc(*args)
        got = cfunc(*args)
        if expected.dtype == np.dtype('int32') and got.dtype == np.dtype('int64'):
            expected = expected.astype(got.dtype)
        self.assertPreciseEqual(expected, got, prec='double', ulps=5)
    args = scalar_args + (None,)
    expected = pyfunc(*args)
    got = cfunc(*args)
    self.assertPreciseEqual(expected, got, prec='double', ulps=5)