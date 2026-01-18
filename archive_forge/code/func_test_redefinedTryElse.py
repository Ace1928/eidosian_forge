from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_redefinedTryElse(self):
    self.flakes('\n        try:\n            from aa import mixer\n        except ImportError:\n            pass\n        else:\n            from bb import mixer\n        mixer(123)\n        ', m.RedefinedWhileUnused)