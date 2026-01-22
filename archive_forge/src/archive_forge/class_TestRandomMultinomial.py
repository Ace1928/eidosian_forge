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
class TestRandomMultinomial(BaseTest):
    """
    Test np.random.multinomial.
    """
    pvals = np.array([1, 1, 1, 2, 3, 1], dtype=np.float64)
    pvals /= pvals.sum()

    def _check_sample(self, n, pvals, sample):
        """
        Check distribution of some samples.
        """
        self.assertIsInstance(sample, np.ndarray)
        self.assertEqual(sample.shape, (len(pvals),))
        self.assertIn(sample.dtype, (np.dtype('int32'), np.dtype('int64')))
        self.assertEqual(sample.sum(), n)
        for p, nexp in zip(pvals, sample):
            self.assertGreaterEqual(nexp, 0)
            self.assertLessEqual(nexp, n)
            pexp = float(nexp) / n
            self.assertGreaterEqual(pexp, p * 0.5)
            self.assertLessEqual(pexp, p * 2.0)

    def test_multinomial_2(self):
        """
        Test multinomial(n, pvals)
        """
        cfunc = jit(nopython=True)(numpy_multinomial2)
        n, pvals = (1000, self.pvals)
        res = cfunc(n, pvals)
        self._check_sample(n, pvals, res)
        pvals = list(pvals)
        res = cfunc(n, pvals)
        self._check_sample(n, pvals, res)
        n = 1000000
        pvals = np.array([1, 0, n // 100, 1], dtype=np.float64)
        pvals /= pvals.sum()
        res = cfunc(n, pvals)
        self._check_sample(n, pvals, res)

    def test_multinomial_3_int(self):
        """
        Test multinomial(n, pvals, size: int)
        """
        cfunc = jit(nopython=True)(numpy_multinomial3)
        n, pvals = (1000, self.pvals)
        k = 10
        res = cfunc(n, pvals, k)
        self.assertEqual(res.shape[0], k)
        for sample in res:
            self._check_sample(n, pvals, sample)

    def test_multinomial_3_tuple(self):
        """
        Test multinomial(n, pvals, size: tuple)
        """
        cfunc = jit(nopython=True)(numpy_multinomial3)
        n, pvals = (1000, self.pvals)
        k = (3, 4)
        res = cfunc(n, pvals, k)
        self.assertEqual(res.shape[:-1], k)
        for sample in res.reshape((-1, res.shape[-1])):
            self._check_sample(n, pvals, sample)