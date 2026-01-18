from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_redefinedClassFunction(self):
    """
        Test that shadowing a function definition in a class suite with another
        one raises a warning.
        """
    self.flakes('\n        class A:\n            def a(): pass\n            def a(): pass\n        ', m.RedefinedWhileUnused)