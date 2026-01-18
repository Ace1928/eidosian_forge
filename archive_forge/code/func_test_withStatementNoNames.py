from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_withStatementNoNames(self):
    """
        No warnings are emitted for using inside or after a nameless C{with}
        statement a name defined beforehand.
        """
    self.flakes('\n        bar = None\n        with open("foo"):\n            bar\n        bar\n        ')