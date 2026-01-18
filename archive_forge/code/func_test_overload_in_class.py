from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_overload_in_class(self):
    self.flakes('\n        from typing import overload\n\n        class C:\n            @overload\n            def f(self, x: int) -> int:\n                pass\n\n            @overload\n            def f(self, x: str) -> str:\n                pass\n\n            def f(self, x): return x\n        ')