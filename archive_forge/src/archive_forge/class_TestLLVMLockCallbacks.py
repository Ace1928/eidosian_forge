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
class TestLLVMLockCallbacks(BaseTest):

    def test_lock_callbacks(self):
        events = []

        def acq():
            events.append('acq')

        def rel():
            events.append('rel')
        llvm.ffi.register_lock_callback(acq, rel)
        self.assertFalse(events)
        llvm.create_module_pass_manager()
        self.assertIn('acq', events)
        self.assertIn('rel', events)
        llvm.ffi.unregister_lock_callback(acq, rel)
        with self.assertRaises(ValueError):
            llvm.ffi.unregister_lock_callback(acq, rel)