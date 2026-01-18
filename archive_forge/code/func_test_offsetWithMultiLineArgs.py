import textwrap
from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.test_other import Test as TestOther
from pyflakes.test.test_imports import Test as TestImports
from pyflakes.test.test_undefined_names import Test as TestUndefinedNames
from pyflakes.test.harness import TestCase, skip
def test_offsetWithMultiLineArgs(self):
    exc1, exc2 = self.flakes('\n            def doctest_stuff(arg1,\n                              arg2,\n                              arg3):\n                """\n                    >>> assert\n                    >>> this\n                """\n            ', m.DoctestSyntaxError, m.UndefinedName).messages
    self.assertEqual(exc1.lineno, 6)
    self.assertEqual(exc1.col, 19)
    self.assertEqual(exc2.lineno, 7)
    self.assertEqual(exc2.col, 12)