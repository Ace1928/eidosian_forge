from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_typing_guard_for_protocol(self):
    self.flakes('\n            from typing import TYPE_CHECKING\n\n            if TYPE_CHECKING:\n                from typing import Protocol\n            else:\n                Protocol = object\n\n            class C(Protocol):\n                def f() -> int:\n                    pass\n        ')