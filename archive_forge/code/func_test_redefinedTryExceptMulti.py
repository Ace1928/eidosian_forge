from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_redefinedTryExceptMulti(self):
    self.flakes('\n        try:\n            from aa import mixer\n        except AttributeError:\n            from bb import mixer\n        except RuntimeError:\n            from cc import mixer\n        except:\n            from dd import mixer\n        mixer(123)\n        ')