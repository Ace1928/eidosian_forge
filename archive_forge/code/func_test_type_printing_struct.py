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
def test_type_printing_struct(self):
    mod = self.module()
    st = mod.get_global_variable('glob_struct')
    self.assertTrue(st.type.is_pointer)
    self.assertIsNotNone(re.match('%struct\\.glob_type(\\.[\\d]+)?\\*', str(st.type)))
    self.assertIsNotNone(re.match('%struct\\.glob_type(\\.[\\d]+)? = type { i64, \\[2 x i64\\] }', str(st.type.element_type)))