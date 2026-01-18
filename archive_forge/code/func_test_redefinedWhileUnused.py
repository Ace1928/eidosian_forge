from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_redefinedWhileUnused(self):
    self.flakes('import fu; fu = 3', m.RedefinedWhileUnused)
    self.flakes('import fu; fu, bar = 3', m.RedefinedWhileUnused)
    self.flakes('import fu; [fu, bar] = 3', m.RedefinedWhileUnused)