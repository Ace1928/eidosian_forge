from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
@skipIf(version_info < (3, 12), 'new in Python 3.12')
def test_type_statements(self):
    self.flakes('\n            type ListOrSet[T] = list[T] | set[T]\n\n            def f(x: ListOrSet[str]) -> None: ...\n\n            type RecursiveType = int | list[RecursiveType]\n\n            type ForwardRef = int | C\n\n            type ForwardRefInBounds[T: C] = T\n\n            class C: pass\n        ')