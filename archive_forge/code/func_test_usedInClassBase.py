from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInClassBase(self):
    self.flakes('\n        import fu\n        class bar(object, fu.baz):\n            pass\n        ')