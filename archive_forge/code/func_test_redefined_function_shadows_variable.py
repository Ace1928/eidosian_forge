from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_redefined_function_shadows_variable(self):
    self.flakes('\n        x = 1\n        def x(): pass\n        ', m.RedefinedWhileUnused)