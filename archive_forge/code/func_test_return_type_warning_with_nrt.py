import os
import subprocess
import sys
import warnings
import numpy as np
import unittest
from numba import jit
from numba.core.errors import (
from numba.core import errors
from numba.tests.support import ignore_internal_warnings
def test_return_type_warning_with_nrt(self):
    """
        Rerun test_return_type_warning with nrt
        """
    y = np.ones(4, dtype=np.float32)

    def return_external_array():
        return y
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', NumbaWarning)
        ignore_internal_warnings()
        cfunc = jit(nopython=True)(return_external_array)
        cfunc()
        self.assertEqual(len(w), 0)