from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_varAugmentedAssignment(self):
    """
        Augmented assignment of a variable is supported.
        We don't care about var refs.
        """
    self.flakes('\n        foo = 0\n        foo += 1\n        ')