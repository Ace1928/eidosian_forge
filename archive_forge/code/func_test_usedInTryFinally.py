from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInTryFinally(self):
    self.flakes('\n        import fu\n        try: pass\n        finally: fu\n        ')
    self.flakes('\n        import fu\n        try: fu\n        finally: pass\n        ')