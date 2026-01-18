from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInAssignment(self):
    self.flakes('import fu; bar=fu')
    self.flakes('import fu; n=0; n+=fu')