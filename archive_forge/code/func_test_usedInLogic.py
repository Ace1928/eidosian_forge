from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInLogic(self):
    self.flakes('import fu; fu and False')
    self.flakes('import fu; fu or False')
    self.flakes('import fu; not fu.bar')