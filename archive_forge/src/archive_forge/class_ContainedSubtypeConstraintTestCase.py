import sys
from tests.base import BaseTestCase
from pyasn1.type import constraint
from pyasn1.type import error
class ContainedSubtypeConstraintTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.c1 = constraint.ContainedSubtypeConstraint(constraint.SingleValueConstraint(12))

    def testGoodVal(self):
        try:
            self.c1(12)
        except error.ValueConstraintError:
            assert 0, 'constraint check fails'

    def testBadVal(self):
        try:
            self.c1(4)
        except error.ValueConstraintError:
            pass
        else:
            assert 0, 'constraint check fails'