from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_notUsedInNestedScope(self):
    self.flakes('\n        import fu\n        def bleh():\n            pass\n        print(fu)\n        ')