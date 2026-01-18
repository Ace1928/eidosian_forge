from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import unittest
from pasta.base import test_utils
def test_simple_function_def(self):
    code = 'def foo(x):\n  return x + 1\n'
    t = ast.parse(code)
    self.checkAstsEqual(t, t)