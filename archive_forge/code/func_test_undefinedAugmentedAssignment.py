import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_undefinedAugmentedAssignment(self):
    self.flakes('\n            def f(seq):\n                a = 0\n                seq[a] += 1\n                seq[b] /= 2\n                c[0] *= 2\n                a -= 3\n                d += 4\n                e[any] = 5\n            ', m.UndefinedName, m.UndefinedName, m.UndefinedName, m.UnusedVariable, m.UndefinedName)