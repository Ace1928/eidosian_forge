import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_undefinedInGenExpNested(self):
    """
        The loop variables of generator expressions nested together are
        not defined in the other generator.
        """
    self.flakes('(b for b in (a for a in [1, 2, 3] if b) if b)', m.UndefinedName)
    self.flakes('(b for b in (a for a in [1, 2, 3] if a) if a)', m.UndefinedName)