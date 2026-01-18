from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_unusedVariableNoLocals(self):
    """
        Using locals() in wrong scope should not matter
        """
    self.flakes('\n        def a():\n            locals()\n            def a():\n                b = 1\n                return\n        ', m.UnusedVariable)