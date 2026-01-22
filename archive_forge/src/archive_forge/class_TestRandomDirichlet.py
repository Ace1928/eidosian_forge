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
class TestRandomDirichlet(BaseTest):
    alpha = np.array([1, 1, 1, 2], dtype=np.float64)

    def _check_sample(self, alpha, size, sample):
        """Check output structure"""
        self.assertIsInstance(sample, np.ndarray)
        self.assertEqual(sample.dtype, np.float64)
        if size is None:
            self.assertEqual(sample.size, len(alpha))
        elif type(size) is int:
            self.assertEqual(sample.shape, (size, len(alpha)))
        else:
            self.assertEqual(sample.shape, size + (len(alpha),))
        'Check statistical properties'
        for val in np.nditer(sample):
            self.assertGreaterEqual(val, 0)
            self.assertLessEqual(val, 1)
        if size is None:
            self.assertAlmostEqual(sample.sum(), 1, places=5)
        else:
            for totals in np.nditer(sample.sum(axis=-1)):
                self.assertAlmostEqual(totals, 1, places=5)

    def test_dirichlet_default(self):
        """
        Test dirichlet(alpha, size=None)
        """
        cfunc = jit(nopython=True)(numpy_dirichlet_default)
        alphas = (self.alpha, tuple(self.alpha), np.array([1, 1, 10000, 1], dtype=np.float64), np.array([1, 1, 1.5, 1], dtype=np.float64))
        for alpha in alphas:
            res = cfunc(alpha)
            self._check_sample(alpha, None, res)

    def test_dirichlet(self):
        """
        Test dirichlet(alpha, size=None)
        """
        cfunc = jit(nopython=True)(numpy_dirichlet)
        sizes = (None, (10,), (10, 10))
        alphas = (self.alpha, tuple(self.alpha), np.array([1, 1, 10000, 1], dtype=np.float64), np.array([1, 1, 1.5, 1], dtype=np.float64))
        for alpha, size in itertools.product(alphas, sizes):
            res = cfunc(alpha, size)
            self._check_sample(alpha, size, res)

    def test_dirichlet_exceptions(self):
        cfunc = jit(nopython=True)(numpy_dirichlet)
        alpha = tuple((0, 1, 1))
        with self.assertRaises(ValueError) as raises:
            cfunc(alpha, 1)
        self.assertIn('dirichlet: alpha must be > 0.0', str(raises.exception))
        alpha = self.alpha
        sizes = (True, 3j, 1.5, (1.5, 1), (3j, 1), (3j, 3j), (np.int8(3), np.int64(7)))
        for size in sizes:
            with self.assertRaises(TypingError) as raises:
                cfunc(alpha, size)
            self.assertIn('np.random.dirichlet(): size should be int or tuple of ints or None, got', str(raises.exception))