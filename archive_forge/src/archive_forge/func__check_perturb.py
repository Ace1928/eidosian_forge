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
def _check_perturb(self, ptr):
    states = []
    for i in range(10):
        _helperlib.rnd_seed(ptr, 0)
        _helperlib.rnd_seed(ptr, os.urandom(512))
        states.append(tuple(_helperlib.rnd_get_state(ptr)[1]))
    self.assertEqual(len(set(states)), len(states))