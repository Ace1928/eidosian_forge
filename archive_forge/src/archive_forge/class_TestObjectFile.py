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
class TestObjectFile(BaseTest):
    mod_asm = '\n        ;ModuleID = <string>\n        target triple = "{triple}"\n\n        declare i32 @sum(i32 %.1, i32 %.2)\n\n        define i32 @sum_twice(i32 %.1, i32 %.2) {{\n            %.3 = call i32 @sum(i32 %.1, i32 %.2)\n            %.4 = call i32 @sum(i32 %.3, i32 %.3)\n            ret i32 %.4\n        }}\n    '

    def test_object_file(self):
        target_machine = self.target_machine(jit=False)
        mod = self.module()
        obj_bin = target_machine.emit_object(mod)
        obj = llvm.ObjectFileRef.from_data(obj_bin)
        has_text = False
        last_address = -1
        for s in obj.sections():
            if s.is_text():
                has_text = True
                self.assertIsNotNone(s.name())
                self.assertTrue(s.size() > 0)
                self.assertTrue(len(s.data()) > 0)
                self.assertIsNotNone(s.address())
                self.assertTrue(last_address < s.address())
                last_address = s.address()
                break
        self.assertTrue(has_text)

    def test_add_object_file(self):
        target_machine = self.target_machine(jit=False)
        mod = self.module()
        obj_bin = target_machine.emit_object(mod)
        obj = llvm.ObjectFileRef.from_data(obj_bin)
        jit = llvm.create_mcjit_compiler(self.module(self.mod_asm), target_machine)
        jit.add_object_file(obj)
        sum_twice = CFUNCTYPE(c_int, c_int, c_int)(jit.get_function_address('sum_twice'))
        self.assertEqual(sum_twice(2, 3), 10)

    def test_add_object_file_from_filesystem(self):
        target_machine = self.target_machine(jit=False)
        mod = self.module()
        obj_bin = target_machine.emit_object(mod)
        temp_desc, temp_path = mkstemp()
        try:
            try:
                f = os.fdopen(temp_desc, 'wb')
                f.write(obj_bin)
                f.flush()
            finally:
                f.close()
            jit = llvm.create_mcjit_compiler(self.module(self.mod_asm), target_machine)
            jit.add_object_file(temp_path)
        finally:
            os.unlink(temp_path)
        sum_twice = CFUNCTYPE(c_int, c_int, c_int)(jit.get_function_address('sum_twice'))
        self.assertEqual(sum_twice(2, 3), 10)

    def test_get_section_content(self):
        elf = bytes.fromhex(issue_632_elf)
        obj = llvm.ObjectFileRef.from_data(elf)
        for s in obj.sections():
            if s.is_text():
                self.assertEqual(len(s.data()), 31)
                self.assertEqual(s.data().hex(), issue_632_text)