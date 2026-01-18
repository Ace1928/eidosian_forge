from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInElse(self):
    self.flakes('\n        import fu\n        if False: pass\n        else: print(fu)\n        ')