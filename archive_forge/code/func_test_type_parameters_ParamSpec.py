from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
@skipIf(version_info < (3, 12), 'new in Python 3.12')
def test_type_parameters_ParamSpec(self):
    self.flakes('\n        from typing import Callable\n\n        def f[R, **P](f: Callable[P, R]) -> Callable[P, R]:\n            def g(*args: P.args, **kwargs: P.kwargs) -> R:\n                return f(*args, **kwargs)\n            return g\n        ')