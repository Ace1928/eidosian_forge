from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_withStatementSingleNameUndefined(self):
    """
        An undefined name warning is emitted if the name first defined by a
        C{with} statement is used before the C{with} statement.
        """
    self.flakes("\n        bar\n        with open('foo') as bar:\n            pass\n        ", m.UndefinedName)