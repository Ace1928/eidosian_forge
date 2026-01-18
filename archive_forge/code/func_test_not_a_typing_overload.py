from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_not_a_typing_overload(self):
    """regression test for @typing.overload detection bug in 2.1.0"""
    self.flakes('\n            def foo(x):\n                return x\n\n            @foo\n            def bar():\n                pass\n\n            def bar():\n                pass\n        ', m.RedefinedWhileUnused)