from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_unusedPackageImport(self):
    """
        If a dotted name is imported and not used, an unused import warning is
        reported.
        """
    self.flakes('import fu.bar', m.UnusedImport)