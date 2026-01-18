import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_undefinedExceptionName(self):
    """Exception names can't be used after the except: block.

        The exc variable is unused inside the exception handler."""
    self.flakes("\n        try:\n            raise ValueError('ve')\n        except ValueError as exc:\n            pass\n        exc\n        ", m.UndefinedName, m.UnusedVariable)