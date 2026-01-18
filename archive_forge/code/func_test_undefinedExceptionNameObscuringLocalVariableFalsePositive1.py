import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_undefinedExceptionNameObscuringLocalVariableFalsePositive1(self):
    """Exception names obscure locals, can't be used after. Unless.

        Last line will never raise UnboundLocalError because it's only
        entered if no exception was raised."""
    self.flakes("\n        exc = 'Original value'\n        try:\n            raise ValueError('ve')\n        except ValueError as exc:\n            print('exception logged')\n            raise\n        exc\n        ", m.UnusedVariable)