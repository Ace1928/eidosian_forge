from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_unusedVariable(self):
    """
        Warn when a variable in a function is assigned a value that's never
        used.
        """
    self.flakes('\n        def a():\n            b = 1\n        ', m.UnusedVariable)