from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_redefinedFunction(self):
    """
        Test that shadowing a function definition with another one raises a
        warning.
        """
    self.flakes('\n        def a(): pass\n        def a(): pass\n        ', m.RedefinedWhileUnused)