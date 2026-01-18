from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_redefinedIfFunction(self):
    """
        Test that shadowing a function definition within an if block
        raises a warning.
        """
    self.flakes('\n        if True:\n            def a(): pass\n            def a(): pass\n        ', m.RedefinedWhileUnused)