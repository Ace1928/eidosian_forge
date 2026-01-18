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
def test_numpy_wald_kwargs(self):
    numba_version = jit_with_kwargs('np.random.wald', ['mean', 'scale'])
    self._check_any_distrib_kwargs(numba_version, get_np_state_ptr(), distrib='wald', paramlist=[{'mean': 1.0, 'scale': 1.0}, {'mean': 2.0, 'scale': 5.0}])
    self.assertRaises(ValueError, numba_version, 0.0, 1.0)
    self.assertRaises(ValueError, numba_version, -0.1, 1.0)
    self.assertRaises(ValueError, numba_version, 1.0, 0.0)
    self.assertRaises(ValueError, numba_version, 1.0, -0.1)