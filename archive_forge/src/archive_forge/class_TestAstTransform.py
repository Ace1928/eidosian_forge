import asyncio
import ast
import os
import signal
import shutil
import sys
import tempfile
import unittest
import pytest
from unittest import mock
from os.path import join
from IPython.core.error import InputRejected
from IPython.core.inputtransformer import InputTransformer
from IPython.core import interactiveshell
from IPython.core.oinspect import OInfo
from IPython.testing.decorators import (
from IPython.testing import tools as tt
from IPython.utils.process import find_cmd
import warnings
import warnings
class TestAstTransform(unittest.TestCase):

    def setUp(self):
        self.negator = Negator()
        ip.ast_transformers.append(self.negator)

    def tearDown(self):
        ip.ast_transformers.remove(self.negator)

    def test_non_int_const(self):
        with tt.AssertPrints('hello'):
            ip.run_cell('print("hello")')

    def test_run_cell(self):
        with tt.AssertPrints('-34'):
            ip.run_cell('print(12 + 22)')
        ip.user_ns['n'] = 55
        with tt.AssertNotPrints('-55'):
            ip.run_cell('print(n)')

    def test_timeit(self):
        called = set()

        def f(x):
            called.add(x)
        ip.push({'f': f})
        with tt.AssertPrints('std. dev. of'):
            ip.run_line_magic('timeit', '-n1 f(1)')
        self.assertEqual(called, {-1})
        called.clear()
        with tt.AssertPrints('std. dev. of'):
            ip.run_cell_magic('timeit', '-n1 f(2)', 'f(3)')
        self.assertEqual(called, {-2, -3})

    def test_time(self):
        called = []

        def f(x):
            called.append(x)
        ip.push({'f': f})
        with tt.AssertPrints('Wall time: '):
            ip.run_line_magic('time', 'f(5+9)')
        self.assertEqual(called, [-14])
        called[:] = []
        with tt.AssertPrints('Wall time: '):
            ip.run_line_magic('time', 'a = f(-3 + -2)')
        self.assertEqual(called, [5])

    def test_macro(self):
        ip.push({'a': 10})
        ip.define_macro('amacro', 'a+=1\nprint(a)')
        with tt.AssertPrints('9'):
            ip.run_cell('amacro')
        with tt.AssertPrints('8'):
            ip.run_cell('amacro')