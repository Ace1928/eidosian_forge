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
def test_object_cache_getbuffer(self):
    notifies = []
    getbuffers = []

    def notify(mod, buf):
        notifies.append((mod, buf))

    def getbuffer(mod):
        getbuffers.append(mod)
    mod = self.module()
    ee = self.jit(mod)
    ee.set_object_cache(notify, getbuffer)
    self.assertEqual(len(notifies), 0)
    self.assertEqual(len(getbuffers), 0)
    cfunc = self.get_sum(ee)
    self.assertEqual(len(notifies), 1)
    self.assertEqual(len(getbuffers), 1)
    self.assertIs(getbuffers[0], mod)
    sum_buffer = notifies[0][1]

    def getbuffer_successful(mod):
        getbuffers.append(mod)
        return sum_buffer
    notifies[:] = []
    getbuffers[:] = []
    mod = self.module(asm_mul)
    ee = self.jit(mod)
    ee.set_object_cache(notify, getbuffer_successful)
    self.assertEqual(len(notifies), 0)
    self.assertEqual(len(getbuffers), 0)
    cfunc = self.get_sum(ee)
    self.assertEqual(cfunc(2, -5), -3)
    self.assertEqual(len(notifies), 0)
    self.assertEqual(len(getbuffers), 1)