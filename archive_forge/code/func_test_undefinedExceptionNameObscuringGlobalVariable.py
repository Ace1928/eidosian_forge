import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
@skip('error reporting disabled due to false positives below')
def test_undefinedExceptionNameObscuringGlobalVariable(self):
    """Exception names obscure globals, can't be used after.

        Last line will raise UnboundLocalError because the existence of that
        exception name creates a local scope placeholder for it, obscuring any
        globals, etc."""
    self.flakes("\n        exc = 'Original value'\n        def func():\n            try:\n                pass  # nothing is raised\n            except ValueError as exc:\n                pass  # block never entered, exc stays unbound\n            exc\n        ", m.UndefinedLocal)