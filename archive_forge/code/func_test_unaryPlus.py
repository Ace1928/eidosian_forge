from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_unaryPlus(self):
    """Don't die on unary +."""
    self.flakes('+1')