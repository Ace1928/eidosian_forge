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
class TestNdOnesLike(TestNdZerosLike):

    def setUp(self):
        super(TestNdOnesLike, self).setUp()
        self.pyfunc = np.ones_like
        self.expected_value = 1

    @unittest.expectedFailure
    def test_like_structured(self):
        super(TestNdOnesLike, self).test_like_structured()

    @unittest.expectedFailure
    def test_like_dtype_structured(self):
        super(TestNdOnesLike, self).test_like_dtype_structured()