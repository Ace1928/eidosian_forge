from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInAssert(self):
    self.flakes('import fu; assert fu.bar')