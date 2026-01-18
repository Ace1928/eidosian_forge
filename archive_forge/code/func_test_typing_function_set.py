from __future__ import absolute_import
import sys
import os.path
from textwrap import dedent
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from Cython.Compiler.ParseTreeTransforms import NormalizeTree, InterpretCompilerDirectives
from Cython.Compiler import Main, Symtab, Visitor, Options
from Cython.TestUtils import TransformTest
def test_typing_function_set(self):
    code = '        def func(x):\n            a = set()\n            # b = {i for i in range(10)} # jedi does not support set comprehension yet\n            c = a\n            d = a | b\n\n        print(func({1,2,3}))\n        '
    types = self._test(code)
    self.assertIn(('func', (1, 0)), types)
    variables = types.pop(('func', (1, 0)))
    self.assertFalse(types)
    self.assertEqual({'a': set(['set']), 'c': set(['set']), 'd': set(['set']), 'x': set(['set'])}, variables)