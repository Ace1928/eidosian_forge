from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_redefinedTryExceptElse(self):
    self.flakes('\n        try:\n            import funca\n        except ImportError:\n            from bb import funca\n            from bb import funcb\n        else:\n            from bbb import funcb\n        print(funca, funcb)\n        ')