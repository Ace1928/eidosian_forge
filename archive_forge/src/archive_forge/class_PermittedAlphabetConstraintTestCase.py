import sys
from tests.base import BaseTestCase
from pyasn1.type import constraint
from pyasn1.type import error
class PermittedAlphabetConstraintTestCase(SingleValueConstraintTestCase):

    def setUp(self):
        self.c1 = constraint.PermittedAlphabetConstraint('A', 'B', 'C')
        self.c2 = constraint.PermittedAlphabetConstraint('DEF')

    def testGoodVal(self):
        try:
            self.c1('A')
        except error.ValueConstraintError:
            assert 0, 'constraint check fails'

    def testBadVal(self):
        try:
            self.c1('E')
        except error.ValueConstraintError:
            pass
        else:
            assert 0, 'constraint check fails'