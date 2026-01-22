import contextlib
import sys
import numpy as np
import random
import re
import threading
import gc
from numba.core.errors import TypingError
from numba import njit
from numba.core import types, utils, config
from numba.tests.support import MemoryLeakMixin, TestCase, tag, skip_if_32bit
import unittest
class TestNdEye(BaseTest):

    def test_eye_n(self):

        def func(n):
            return np.eye(n)
        self.check_outputs(func, [(1,), (3,)])

    def test_eye_n_dtype(self):
        for dt in (None, np.complex128, np.complex64(1)):

            def func(n, dtype=dt):
                return np.eye(n, dtype=dtype)
            self.check_outputs(func, [(1,), (3,)])

    def test_eye_n_m(self):

        def func(n, m):
            return np.eye(n, m)
        self.check_outputs(func, [(1, 2), (3, 2), (0, 3)])

    def check_eye_n_m_k(self, func):
        self.check_outputs(func, [(1, 2, 0), (3, 4, 1), (3, 4, -1), (4, 3, -2), (4, 3, -5), (4, 3, 5)])

    def test_eye_n_m_k(self):

        def func(n, m, k):
            return np.eye(n, m, k)
        self.check_eye_n_m_k(func)

    def test_eye_n_m_k_dtype(self):

        def func(n, m, k):
            return np.eye(N=n, M=m, k=k, dtype=np.int16)
        self.check_eye_n_m_k(func)

    def test_eye_n_m_k_dtype_instance(self):
        dtype = np.dtype('int16')

        def func(n, m, k):
            return np.eye(N=n, M=m, k=k, dtype=dtype)
        self.check_eye_n_m_k(func)