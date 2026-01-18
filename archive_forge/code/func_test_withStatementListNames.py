from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_withStatementListNames(self):
    """
        No warnings are emitted for using any of the list of names defined by a
        C{with} statement within the suite or afterwards.
        """
    self.flakes("\n        with open('foo') as [bar, baz]:\n            bar, baz\n        bar, baz\n        ")