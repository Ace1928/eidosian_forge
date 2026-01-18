from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInFunction(self):
    self.flakes('\n        import fu\n        def fun():\n            print(fu)\n        ')