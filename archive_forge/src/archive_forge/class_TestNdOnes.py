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
class TestNdOnes(TestNdZeros):

    def setUp(self):
        super(TestNdOnes, self).setUp()
        self.pyfunc = np.ones

    @unittest.expectedFailure
    def test_1d_dtype_str_structured_dtype(self):
        super().test_1d_dtype_str_structured_dtype()