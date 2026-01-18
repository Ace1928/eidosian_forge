from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_shadowedByLambda(self):
    self.flakes('import fu; lambda fu: fu', m.UnusedImport, m.RedefinedWhileUnused)
    self.flakes('import fu; lambda fu: fu\nfu()')