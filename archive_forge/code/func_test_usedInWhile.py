from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInWhile(self):
    self.flakes('\n        import fu\n        while 0:\n            fu\n        ')
    self.flakes('\n        import fu\n        while fu: pass\n        ')