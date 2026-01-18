from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_unusedImport_underscore(self):
    """
        The magic underscore var should be reported as unused when used as an
        import alias.
        """
    self.flakes('import fu as _', m.UnusedImport)