import ctypes
import threading
from ctypes import CFUNCTYPE, c_int, c_int32
from ctypes.util import find_library
import gc
import locale
import os
import platform
import re
import subprocess
import sys
import unittest
from contextlib import contextmanager
from tempfile import mkstemp
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.binding import ffi
from llvmlite.tests import TestCase
def test_target_data_abi_enquiries(self):
    mod = self.module()
    ee = self.jit(mod)
    td = ee.target_data
    gv_i32 = mod.get_global_variable('glob')
    gv_i8 = mod.get_global_variable('glob_b')
    gv_struct = mod.get_global_variable('glob_struct')
    pointer_size = 4 if sys.maxsize < 2 ** 32 else 8
    for g in (gv_i32, gv_i8, gv_struct):
        self.assertEqual(td.get_abi_size(g.type), pointer_size)
    self.assertEqual(td.get_pointee_abi_size(gv_i32.type), 4)
    self.assertEqual(td.get_pointee_abi_alignment(gv_i32.type), 4)
    self.assertEqual(td.get_pointee_abi_size(gv_i8.type), 1)
    self.assertIn(td.get_pointee_abi_alignment(gv_i8.type), (1, 2, 4))
    self.assertEqual(td.get_pointee_abi_size(gv_struct.type), 24)
    self.assertIn(td.get_pointee_abi_alignment(gv_struct.type), (4, 8))