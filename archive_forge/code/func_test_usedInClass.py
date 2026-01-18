from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInClass(self):
    self.flakes('\n        import fu\n        class bar:\n            bar = fu\n        ')