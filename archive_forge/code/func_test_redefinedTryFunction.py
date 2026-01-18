from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_redefinedTryFunction(self):
    """
        Test that shadowing a function definition within a try block
        raises a warning.
        """
    self.flakes('\n        try:\n            def a(): pass\n            def a(): pass\n        except:\n            pass\n        ', m.RedefinedWhileUnused)