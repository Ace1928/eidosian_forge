from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_shadowedByParameter(self):
    self.flakes('\n        import fu\n        def fun(fu):\n            print(fu)\n        ', m.UnusedImport, m.RedefinedWhileUnused)
    self.flakes('\n        import fu\n        def fun(fu):\n            print(fu)\n        print(fu)\n        ')