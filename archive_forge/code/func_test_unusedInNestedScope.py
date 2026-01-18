from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_unusedInNestedScope(self):
    self.flakes('\n        def bar():\n            import fu\n        fu\n        ', m.UnusedImport, m.UndefinedName)