from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_withStatementSubscript(self):
    """
        No warnings are emitted for using a subscript as the target of a
        C{with} statement.
        """
    self.flakes("\n        import foo\n        with open('foo') as foo[0]:\n            pass\n        ")