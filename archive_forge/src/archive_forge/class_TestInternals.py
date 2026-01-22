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
class TestInternals(BaseTest):
    """
    Test low-level internals of the implementation.
    """

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

    def _check_shuffle(self, ptr):
        r = random.Random()
        ints, index = _copy_py_state(r, ptr)
        for i in range(index, N + 1, 2):
            r.random()
        _helperlib.rnd_shuffle(ptr)
        mt = r.getstate()[1]
        ints, index = (mt[:-1], mt[-1])
        self.assertEqual(_helperlib.rnd_get_state(ptr)[1], list(ints))

    def _check_init(self, ptr):
        r = np.random.RandomState()
        for i in [0, 1, 125, 2 ** 32 - 5]:
            r.seed(np.uint32(i))
            st = r.get_state()
            ints = list(st[1])
            index = st[2]
            assert index == N
            _helperlib.rnd_seed(ptr, i)
            self.assertEqual(_helperlib.rnd_get_state(ptr), (index, ints))

    def _check_perturb(self, ptr):
        states = []
        for i in range(10):
            _helperlib.rnd_seed(ptr, 0)
            _helperlib.rnd_seed(ptr, os.urandom(512))
            states.append(tuple(_helperlib.rnd_get_state(ptr)[1]))
        self.assertEqual(len(set(states)), len(states))

    def test_get_set_state(self):
        self._check_get_set_state(get_py_state_ptr())

    def test_shuffle(self):
        self._check_shuffle(get_py_state_ptr())

    def test_init(self):
        self._check_init(get_py_state_ptr())

    def test_perturb(self):
        self._check_perturb(get_py_state_ptr())