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
def test_populate_function_pass_manager(self):
    mod = self.module()
    pmb = self.pmb()
    pm = llvm.create_function_pass_manager(mod)
    pmb.populate(pm)
    pmb.close()
    pm.close()