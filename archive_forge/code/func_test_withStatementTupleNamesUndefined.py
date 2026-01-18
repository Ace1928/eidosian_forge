from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_withStatementTupleNamesUndefined(self):
    """
        An undefined name warning is emitted if a name first defined by the
        tuple-unpacking form of the C{with} statement is used before the
        C{with} statement.
        """
    self.flakes("\n        baz\n        with open('foo') as (bar, baz):\n            pass\n        ", m.UndefinedName)