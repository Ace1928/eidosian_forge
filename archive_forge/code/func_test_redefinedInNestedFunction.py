from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_redefinedInNestedFunction(self):
    """
        Test that shadowing a global name with a nested function definition
        generates a warning.
        """
    self.flakes('\n        import fu\n        def bar():\n            def baz():\n                def fu():\n                    pass\n        ', m.RedefinedWhileUnused, m.UnusedImport)