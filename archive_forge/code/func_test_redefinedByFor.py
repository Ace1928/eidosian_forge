from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_redefinedByFor(self):
    self.flakes('\n        import fu\n        for fu in range(2):\n            pass\n        ', m.ImportShadowedByLoopVar)