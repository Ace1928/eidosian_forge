from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
@skipIf(version_info < (3, 12), 'new in Python 3.12')
def test_type_parameters_functions(self):
    self.flakes('\n            def f[T](t: T) -> T: return t\n\n            async def g[T](t: T) -> T: return t\n\n            def with_forward_ref[T: C](t: T) -> T: return t\n\n            def can_access_inside[T](t: T) -> T:\n                print(T)\n                return t\n\n            class C: pass\n        ')