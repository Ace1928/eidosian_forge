from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
@skipIf(version_info < (3, 11), 'new in Python 3.11')
def test_variadic_generics(self):
    self.flakes("\n            from typing import Generic\n            from typing import TypeVarTuple\n\n            Ts = TypeVarTuple('Ts')\n\n            class Shape(Generic[*Ts]): pass\n\n            def f(*args: *Ts) -> None: ...\n\n            def g(x: Shape[*Ts]) -> Shape[*Ts]: ...\n        ")