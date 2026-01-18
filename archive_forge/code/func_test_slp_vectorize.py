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
def test_slp_vectorize(self):
    pmb = self.pmb()
    self.assertIsInstance(pmb.slp_vectorize, bool)
    for b in (True, False):
        pmb.slp_vectorize = b
        self.assertEqual(pmb.slp_vectorize, b)