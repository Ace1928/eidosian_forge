import unittest
import inspect
from numba import njit
from numba.tests.support import TestCase
from numba.misc.firstlinefinder import get_func_body_first_lineno
def test_string_eval(self):
    source = 'def foo():\n            pass\n        '
    globalns = {}
    exec(source, globalns)
    foo = globalns['foo']
    first_def_line = get_func_body_first_lineno(foo)
    self.assertIsNone(first_def_line)