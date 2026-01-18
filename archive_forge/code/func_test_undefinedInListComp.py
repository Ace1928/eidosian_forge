import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_undefinedInListComp(self):
    self.flakes('\n        [a for a in range(10)]\n        a\n        ', m.UndefinedName)