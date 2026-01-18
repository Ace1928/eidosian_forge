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
def test_noncentral_chisquare_default(self):
    """
        Test noncentral_chisquare(df, nonc, size=None)
        """
    cfunc = jit(nopython=True)(numpy_noncentral_chisquare_default)
    inputs = ((0.5, 1), (1, 5), (5, 1), (100000, 1), (1, 10000))
    for df, nonc in inputs:
        res = cfunc(df, nonc)
        self._check_sample(None, res)
        res = cfunc(df, np.nan)
        self.assertTrue(np.isnan(res))