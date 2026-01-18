from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_redefinedInSetComprehension(self):
    """
        Test that reusing a variable in a set comprehension does not raise
        a warning.
        """
    self.flakes('\n        a = 1\n        {1 for a, b in [(1, 2)]}\n        ')
    self.flakes('\n        class A:\n            a = 1\n            {1 for a, b in [(1, 2)]}\n        ')
    self.flakes('\n        def f():\n            a = 1\n            {1 for a, b in [(1, 2)]}\n        ', m.UnusedVariable)
    self.flakes('\n        {1 for a, b in [(1, 2)]}\n        {1 for a, b in [(1, 2)]}\n        ')
    self.flakes('\n        for a, b in [(1, 2)]:\n            pass\n        {1 for a, b in [(1, 2)]}\n        ')