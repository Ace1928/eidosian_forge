from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_withStatementSubscriptUndefined(self):
    """
        An undefined name warning is emitted if the subscript used as the
        target of a C{with} statement is not defined.
        """
    self.flakes("\n        import foo\n        with open('foo') as foo[bar]:\n            pass\n        ", m.UndefinedName)