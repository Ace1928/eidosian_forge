from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_setComprehensionAndLiteral(self):
    """
        Set comprehensions are properly handled.
        """
    self.flakes('\n        a = {1, 2, 3}\n        b = {x for x in range(10)}\n        ')