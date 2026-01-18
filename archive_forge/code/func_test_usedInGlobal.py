from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInGlobal(self):
    """
        A 'global' statement shadowing an unused import should not prevent it
        from being reported.
        """
    self.flakes('\n        import fu\n        def f(): global fu\n        ', m.UnusedImport)