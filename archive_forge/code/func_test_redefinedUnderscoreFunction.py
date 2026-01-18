from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_redefinedUnderscoreFunction(self):
    """
        Test that shadowing a function definition named with underscore doesn't
        raise anything.
        """
    self.flakes('\n        def _(): pass\n        def _(): pass\n        ')