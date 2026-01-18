import numpy as np
import unittest
from numba import njit
from numba.core.errors import TypingError
from numba import jit, typeof
from numba.core import types
from numba.tests.support import TestCase
def test_record_array_const_contig(self):
    self.check_array_const(getitem4)