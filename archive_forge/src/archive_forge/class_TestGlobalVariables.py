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
class TestGlobalVariables(BaseTest):

    def check_global_variable_linkage(self, linkage, has_undef=True):
        mod = ir.Module()
        typ = ir.IntType(32)
        gv = ir.GlobalVariable(mod, typ, 'foo')
        gv.linkage = linkage
        asm = str(mod)
        if has_undef:
            self.assertIn('undef', asm)
        else:
            self.assertNotIn('undef', asm)
        self.module(asm)

    def test_internal_linkage(self):
        self.check_global_variable_linkage('internal')

    def test_common_linkage(self):
        self.check_global_variable_linkage('common')

    def test_external_linkage(self):
        self.check_global_variable_linkage('external', has_undef=False)

    def test_available_externally_linkage(self):
        self.check_global_variable_linkage('available_externally')

    def test_private_linkage(self):
        self.check_global_variable_linkage('private')

    def test_linkonce_linkage(self):
        self.check_global_variable_linkage('linkonce')

    def test_weak_linkage(self):
        self.check_global_variable_linkage('weak')

    def test_appending_linkage(self):
        self.check_global_variable_linkage('appending')

    def test_extern_weak_linkage(self):
        self.check_global_variable_linkage('extern_weak', has_undef=False)

    def test_linkonce_odr_linkage(self):
        self.check_global_variable_linkage('linkonce_odr')

    def test_weak_odr_linkage(self):
        self.check_global_variable_linkage('weak_odr')