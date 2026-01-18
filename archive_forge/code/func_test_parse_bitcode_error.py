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
def test_parse_bitcode_error(self):
    with self.assertRaises(RuntimeError) as cm:
        llvm.parse_bitcode(b'')
    self.assertIn('LLVM bitcode parsing error', str(cm.exception))
    if llvm.llvm_version_info[0] < 9:
        self.assertIn('Invalid bitcode signature', str(cm.exception))
    else:
        self.assertIn('file too small to contain bitcode header', str(cm.exception))