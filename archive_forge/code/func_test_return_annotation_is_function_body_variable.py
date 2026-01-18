from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_return_annotation_is_function_body_variable(self):
    self.flakes('\n        class Test:\n            def t(self) -> Y:\n                Y = 2\n                return Y\n        ', m.UndefinedName)