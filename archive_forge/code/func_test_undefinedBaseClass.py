from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_undefinedBaseClass(self):
    """
        If a name in the base list of a class definition is undefined, a
        warning is emitted.
        """
    self.flakes('\n        class foo(foo):\n            pass\n        ', m.UndefinedName)