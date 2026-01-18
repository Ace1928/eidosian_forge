import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def test_raise_invalid_class_objmode(self):
    self.check_raise_invalid_class(int, flags=force_pyobj_flags)
    self.check_raise_invalid_class(1, flags=force_pyobj_flags)