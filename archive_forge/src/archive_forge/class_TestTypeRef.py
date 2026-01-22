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
class TestTypeRef(BaseTest):

    def test_str(self):
        mod = self.module()
        glob = mod.get_global_variable('glob')
        self.assertEqual(str(glob.type), 'i32*')
        glob_struct_type = mod.get_struct_type('struct.glob_type')
        self.assertEqual(str(glob_struct_type), '%struct.glob_type = type { i64, [2 x i64] }')
        elements = list(glob_struct_type.elements)
        self.assertEqual(len(elements), 2)
        self.assertEqual(str(elements[0]), 'i64')
        self.assertEqual(str(elements[1]), '[2 x i64]')

    def test_type_kind(self):
        mod = self.module()
        glob = mod.get_global_variable('glob')
        self.assertEqual(glob.type.type_kind, llvm.TypeKind.pointer)
        self.assertTrue(glob.type.is_pointer)
        glob_struct = mod.get_global_variable('glob_struct')
        self.assertEqual(glob_struct.type.type_kind, llvm.TypeKind.pointer)
        self.assertTrue(glob_struct.type.is_pointer)
        stype = next(iter(glob_struct.type.elements))
        self.assertEqual(stype.type_kind, llvm.TypeKind.struct)
        self.assertTrue(stype.is_struct)
        stype_a, stype_b = stype.elements
        self.assertEqual(stype_a.type_kind, llvm.TypeKind.integer)
        self.assertEqual(stype_b.type_kind, llvm.TypeKind.array)
        self.assertTrue(stype_b.is_array)
        glob_vec_struct_type = mod.get_struct_type('struct.glob_type_vec')
        _, vector_type = glob_vec_struct_type.elements
        self.assertEqual(vector_type.type_kind, llvm.TypeKind.vector)
        self.assertTrue(vector_type.is_vector)
        funcptr = mod.get_function('sum').type
        self.assertEqual(funcptr.type_kind, llvm.TypeKind.pointer)
        functype, = funcptr.elements
        self.assertEqual(functype.type_kind, llvm.TypeKind.function)

    def test_element_count(self):
        mod = self.module()
        glob_struct_type = mod.get_struct_type('struct.glob_type')
        _, array_type = glob_struct_type.elements
        self.assertEqual(array_type.element_count, 2)
        with self.assertRaises(ValueError):
            glob_struct_type.element_count

    def test_type_width(self):
        mod = self.module()
        glob_struct_type = mod.get_struct_type('struct.glob_type')
        glob_vec_struct_type = mod.get_struct_type('struct.glob_type_vec')
        integer_type, array_type = glob_struct_type.elements
        _, vector_type = glob_vec_struct_type.elements
        self.assertEqual(integer_type.type_width, 64)
        self.assertEqual(vector_type.type_width, 64 * 2)
        self.assertEqual(glob_struct_type.type_width, 0)
        self.assertEqual(array_type.type_width, 0)

    def test_vararg_function(self):
        mod = self.module(asm_vararg_declare)
        func = mod.get_function('vararg')
        decltype = func.type.element_type
        self.assertTrue(decltype.is_function_vararg)
        mod = self.module(asm_sum_declare)
        func = mod.get_function('sum')
        decltype = func.type.element_type
        self.assertFalse(decltype.is_function_vararg)
        self.assertTrue(func.type.is_pointer)
        with self.assertRaises(ValueError) as raises:
            func.type.is_function_vararg
        self.assertIn('Type i32 (i32, i32)* is not a function', str(raises.exception))