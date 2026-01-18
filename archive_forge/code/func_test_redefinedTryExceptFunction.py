from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_redefinedTryExceptFunction(self):
    """
        Test that shadowing a function definition twice in try
        and except block does not raise a warning.
        """
    self.flakes('\n        try:\n            def a(): pass\n        except:\n            def a(): pass\n        ')