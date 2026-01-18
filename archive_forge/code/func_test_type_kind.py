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