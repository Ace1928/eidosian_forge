from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_print_function_assignment(self):
    """
        A valid assignment, tested for catching false positives.
        """
    self.flakes('\n        from __future__ import print_function\n        log = print\n        log("Hello")\n        ')