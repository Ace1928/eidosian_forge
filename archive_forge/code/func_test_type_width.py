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
def test_type_width(self):
    mod = self.module()
    glob_struct_type = mod.get_struct_type('struct.glob_type')
    glob_vec_struct_type = mod.get_struct_type('struct.glob_type_vec')
    integer_type, array_type = glob_struct_type.elements
    _, vector_type = glob_vec_struct_type.elements
    self.assertEqual(integer_type.type_width, 64)
    self.assertEqual(vector_type.type_width, 64 * 2)
    self.assertEqual(glob_struct_type.type_width, 0)
    self.assertEqual(array_type.type_width, 0)