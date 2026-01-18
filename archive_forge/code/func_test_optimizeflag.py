import _imp
import importlib.util
import unittest
import sys
from ctypes import *
from test.support import import_helper
import _ctypes_test
def test_optimizeflag(self):
    opt = c_int.in_dll(pythonapi, 'Py_OptimizeFlag').value
    self.assertEqual(opt, sys.flags.optimize)