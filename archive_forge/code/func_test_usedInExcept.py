from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInExcept(self):
    self.flakes('\n        import fu\n        try: fu\n        except: pass\n        ')