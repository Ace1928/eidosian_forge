from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_yieldFromUndefined(self):
    """
        Test C{yield from} statement
        """
    self.flakes('\n        def bar():\n            yield from foo()\n        ', m.UndefinedName)