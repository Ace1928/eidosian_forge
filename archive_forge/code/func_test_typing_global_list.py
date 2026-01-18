from __future__ import absolute_import
import sys
import os.path
from textwrap import dedent
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from Cython.Compiler.ParseTreeTransforms import NormalizeTree, InterpretCompilerDirectives
from Cython.Compiler import Main, Symtab, Visitor, Options
from Cython.TestUtils import TransformTest
def test_typing_global_list(self):
    code = '        a = [x for x in range(10)]\n        b = list(range(10))\n        c = a + b\n        d = [0]*10\n        '
    types = self._test(code)
    self.assertIn((None, (1, 0)), types)
    variables = types.pop((None, (1, 0)))
    self.assertFalse(types)
    self.assertEqual({'a': set(['list']), 'b': set(['list']), 'c': set(['list']), 'd': set(['list'])}, variables)