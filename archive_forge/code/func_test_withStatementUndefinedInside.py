from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_withStatementUndefinedInside(self):
    """
        An undefined name warning is emitted if a name is used inside the
        body of a C{with} statement without first being bound.
        """
    self.flakes("\n        with open('foo') as bar:\n            baz\n        ", m.UndefinedName)