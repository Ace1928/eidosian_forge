from __future__ import absolute_import
import os
import re
import sys
import trace
import inspect
import warnings
import unittest
import textwrap
import tempfile
import functools
import traceback
import itertools
import gdb
from .. import libcython
from .. import libpython
from . import TestLibCython as test_libcython
from ...Utils import add_metaclass
class TestDebugInformationClasses(DebugTestCase):

    def test_CythonModule(self):
        """test that debug information was parsed properly into data structures"""
        self.assertEqual(self.module.name, 'codefile')
        global_vars = ('c_var', 'python_var', '__name__', '__builtins__', '__doc__', '__file__')
        assert set(global_vars).issubset(self.module.globals)

    def test_CythonVariable(self):
        module_globals = self.module.globals
        c_var = module_globals['c_var']
        python_var = module_globals['python_var']
        self.assertEqual(c_var.type, libcython.CObject)
        self.assertEqual(python_var.type, libcython.PythonObject)
        self.assertEqual(c_var.qualified_name, 'codefile.c_var')

    def test_CythonFunction(self):
        self.assertEqual(self.spam_func.qualified_name, 'codefile.spam')
        self.assertEqual(self.spam_meth.qualified_name, 'codefile.SomeClass.spam')
        self.assertEqual(self.spam_func.module, self.module)
        assert self.eggs_func.pf_cname, (self.eggs_func, self.eggs_func.pf_cname)
        assert not self.ham_func.pf_cname
        assert not self.spam_func.pf_cname
        assert not self.spam_meth.pf_cname
        self.assertEqual(self.spam_func.type, libcython.CObject)
        self.assertEqual(self.ham_func.type, libcython.CObject)
        self.assertEqual(self.spam_func.arguments, ['a'])
        self.assertEqual(self.spam_func.step_into_functions, {'puts', 'some_c_function'})
        expected_lineno = test_libcython.source_to_lineno['def spam(a=0):']
        self.assertEqual(self.spam_func.lineno, expected_lineno)
        self.assertEqual(sorted(self.spam_func.locals), list('abcd'))