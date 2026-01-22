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
class JITTestMixin(object):
    """
    Mixin for ExecutionEngine tests.
    """

    def get_sum(self, ee, func_name='sum'):
        ee.finalize_object()
        cfptr = ee.get_function_address(func_name)
        self.assertTrue(cfptr)
        return CFUNCTYPE(c_int, c_int, c_int)(cfptr)

    def test_run_code(self):
        mod = self.module()
        with self.jit(mod) as ee:
            cfunc = self.get_sum(ee)
            res = cfunc(2, -5)
            self.assertEqual(-3, res)

    def test_close(self):
        ee = self.jit(self.module())
        ee.close()
        ee.close()
        with self.assertRaises(ctypes.ArgumentError):
            ee.finalize_object()

    def test_with(self):
        ee = self.jit(self.module())
        with ee:
            pass
        with self.assertRaises(RuntimeError):
            with ee:
                pass
        with self.assertRaises(ctypes.ArgumentError):
            ee.finalize_object()

    def test_module_lifetime(self):
        mod = self.module()
        ee = self.jit(mod)
        ee.close()
        mod.close()

    def test_module_lifetime2(self):
        mod = self.module()
        ee = self.jit(mod)
        mod.close()
        ee.close()

    def test_add_module(self):
        ee = self.jit(self.module())
        mod = self.module(asm_mul)
        ee.add_module(mod)
        with self.assertRaises(KeyError):
            ee.add_module(mod)
        self.assertFalse(mod.closed)
        ee.close()
        self.assertTrue(mod.closed)

    def test_add_module_lifetime(self):
        ee = self.jit(self.module())
        mod = self.module(asm_mul)
        ee.add_module(mod)
        mod.close()
        ee.close()

    def test_add_module_lifetime2(self):
        ee = self.jit(self.module())
        mod = self.module(asm_mul)
        ee.add_module(mod)
        ee.close()
        mod.close()

    def test_remove_module(self):
        ee = self.jit(self.module())
        mod = self.module(asm_mul)
        ee.add_module(mod)
        ee.remove_module(mod)
        with self.assertRaises(KeyError):
            ee.remove_module(mod)
        self.assertFalse(mod.closed)
        ee.close()
        self.assertFalse(mod.closed)

    def test_target_data(self):
        mod = self.module()
        ee = self.jit(mod)
        td = ee.target_data
        self.assertIs(ee.target_data, td)
        str(td)
        del mod, ee
        str(td)

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

    def test_object_cache_notify(self):
        notifies = []

        def notify(mod, buf):
            notifies.append((mod, buf))
        mod = self.module()
        ee = self.jit(mod)
        ee.set_object_cache(notify)
        self.assertEqual(len(notifies), 0)
        cfunc = self.get_sum(ee)
        cfunc(2, -5)
        self.assertEqual(len(notifies), 1)
        self.assertIs(notifies[0][0], mod)
        self.assertIsInstance(notifies[0][1], bytes)
        notifies[:] = []
        mod2 = self.module(asm_mul)
        ee.add_module(mod2)
        cfunc = self.get_sum(ee, 'mul')
        self.assertEqual(len(notifies), 1)
        self.assertIs(notifies[0][0], mod2)
        self.assertIsInstance(notifies[0][1], bytes)

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