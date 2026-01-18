from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInTuple(self):
    self.flakes('import fu; (fu,)')