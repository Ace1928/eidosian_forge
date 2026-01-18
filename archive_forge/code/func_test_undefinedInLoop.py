import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_undefinedInLoop(self):
    """
        The loop variable is defined after the expression is computed.
        """
    self.flakes('\n        for i in range(i):\n            print(i)\n        ', m.UndefinedName)
    self.flakes('\n        [42 for i in range(i)]\n        ', m.UndefinedName)
    self.flakes('\n        (42 for i in range(i))\n        ', m.UndefinedName)