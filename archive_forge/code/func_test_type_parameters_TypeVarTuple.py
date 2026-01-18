from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
@skipIf(version_info < (3, 12), 'new in Python 3.12')
def test_type_parameters_TypeVarTuple(self):
    self.flakes('\n        def f[*T](*args: *T) -> None: ...\n        ')