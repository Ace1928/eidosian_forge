from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
@skipIf(version_info < (3, 12), 'new in Python 3.12')
def test_type_parameters_do_not_escape_function_scopes(self):
    self.flakes('\n            from x import g\n\n            @g(T)  # not accessible in decorators\n            def f[T](t: T) -> T: return t\n\n            T  # not accessible afterwards\n        ', m.UndefinedName, m.UndefinedName)