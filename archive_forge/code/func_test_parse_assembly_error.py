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
def test_parse_assembly_error(self):
    with self.assertRaises(RuntimeError) as cm:
        self.module(asm_parse_error)
    s = str(cm.exception)
    self.assertIn('parsing error', s)
    self.assertIn('invalid operand type', s)