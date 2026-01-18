from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInGetattr(self):
    self.flakes('import fu; fu.bar.baz')
    self.flakes('import fu; "bar".fu.baz', m.UnusedImport)