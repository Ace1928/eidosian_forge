from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInSlice(self):
    self.flakes('import fu; print(fu.bar[1:])')