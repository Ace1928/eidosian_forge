from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInParameterDefault(self):
    self.flakes('\n        import fu\n        def f(bar=fu):\n            pass\n        ')