from __future__ import absolute_import
import sys
import os.path
from textwrap import dedent
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from Cython.Compiler.ParseTreeTransforms import NormalizeTree, InterpretCompilerDirectives
from Cython.Compiler import Main, Symtab, Visitor, Options
from Cython.TestUtils import TransformTest
def test_typing_function_list(self):
    code = '        def func(x):\n            a = [[], []]\n            b = [0]* 10 + a\n            c = a[0]\n\n        print(func([0]*100))\n        '
    types = self._test(code)
    self.assertIn(('func', (1, 0)), types)
    variables = types.pop(('func', (1, 0)))
    self.assertFalse(types)
    self.assertEqual({'a': set(['list']), 'b': set(['list']), 'c': set(['list']), 'x': set(['list'])}, variables)