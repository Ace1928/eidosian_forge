from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_redefinedByGenExp(self):
    """
        Re-using a global name as the loop variable for a generator
        expression results in a redefinition warning.
        """
    self.flakes('import fu; (1 for fu in range(1))', m.RedefinedWhileUnused, m.UnusedImport)