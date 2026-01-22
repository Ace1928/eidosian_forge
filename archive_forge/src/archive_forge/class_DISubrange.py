from numba.tests.support import TestCase, linux_only
from numba.tests.gdb_support import needs_gdb, skip_unless_pexpect, GdbMIDriver
from unittest.mock import patch, Mock
from numba.core import datamodel
import numpy as np
from numba import typeof
import ctypes as ct
import unittest
class DISubrange:

    def __init__(self, lo, hi):
        self._lo = lo
        self._hi = hi

    @property
    def type(self):
        return self

    def range(self):
        return (self._lo, self._hi)