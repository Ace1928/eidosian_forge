from numba.tests.support import TestCase, linux_only
from numba.tests.gdb_support import needs_gdb, skip_unless_pexpect, GdbMIDriver
from unittest.mock import patch, Mock
from numba.core import datamodel
import numpy as np
from numba import typeof
import ctypes as ct
import unittest
def test_np_array_printer_simple_structured_dtype(self):
    n = 4
    m = 3
    aligned = np.dtype([('x', np.int16), ('y', np.float64)], align=True)
    unaligned = np.dtype([('x', np.int16), ('y', np.float64)], align=False)
    for dt in (aligned, unaligned):
        arr = np.empty(m * n, dtype=dt).reshape(m, n)
        arr['x'] = np.arange(m * n, dtype=dt['x']).reshape(m, n)
        arr['y'] = 100 * np.arange(m * n, dtype=dt['y']).reshape(m, n)
        self.check(arr)