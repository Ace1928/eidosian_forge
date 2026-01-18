from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInDict(self):
    self.flakes('import fu; {fu:None}')
    self.flakes('import fu; {1:fu}')