from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_return_annotation_is_class_scope_variable(self):
    self.flakes("\n        from typing import TypeVar\n        class Test:\n            Y = TypeVar('Y')\n\n            def t(self, x: Y) -> Y:\n                return x\n        ")