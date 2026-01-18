import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_unusedAsStarUnpack(self):
    """
        Star names in unpack are unused if RHS is a tuple/list literal.
        """
    self.flakes("\n        def f():\n            a, *b = any, all, 4, 2, 'un'\n        ", m.UnusedVariable, m.UnusedVariable)
    self.flakes('\n        def f():\n            (*a, b) = [bool, int, float, complex]\n        ', m.UnusedVariable, m.UnusedVariable)
    self.flakes('\n        def f():\n            [a, *b, c] = 9, 8, 7, 6, 5, 4\n        ', m.UnusedVariable, m.UnusedVariable, m.UnusedVariable)