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
def _check_get_set_state(self, ptr):
    state = _helperlib.rnd_get_state(ptr)
    i, ints = state
    self.assertIsInstance(i, int)
    self.assertIsInstance(ints, list)
    self.assertEqual(len(ints), N)
    j = i * 100007 % N
    ints = [i * 3 for i in range(N)]
    _helperlib.rnd_set_state(ptr, (j, ints))
    self.assertEqual(_helperlib.rnd_get_state(ptr), (j, ints))