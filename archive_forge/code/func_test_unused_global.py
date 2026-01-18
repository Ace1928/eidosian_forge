import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
@skip('todo')
def test_unused_global(self):
    """An unused global statement does not define the name."""
    self.flakes('\n        def f1():\n            m\n\n        def f2():\n            global m\n        ', m.UndefinedName)