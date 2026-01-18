from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInYield(self):
    self.flakes('\n        import fu\n        def gen():\n            yield fu\n        ')