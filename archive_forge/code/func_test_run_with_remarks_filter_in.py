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
def test_run_with_remarks_filter_in(self):
    mod = self.module(licm_asm)
    fn = mod.get_function('licm')
    pm = self.pm(mod)
    pm.add_licm_pass()
    self.pmb().populate(pm)
    mod.close()
    pm.initialize()
    ok, remarks = pm.run_with_remarks(fn, remarks_filter='licm')
    pm.finalize()
    self.assertTrue(ok)
    self.assertIn('Passed', remarks)
    self.assertIn('licm', remarks)