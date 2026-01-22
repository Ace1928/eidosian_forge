import sys
from tests.base import BaseTestCase
from pyasn1.type import constraint
from pyasn1.type import error
class IndirectDerivationTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.c1 = constraint.ConstraintsIntersection(constraint.ValueRangeConstraint(1, 30))
        self.c2 = constraint.ConstraintsIntersection(self.c1, constraint.ValueRangeConstraint(1, 20))
        self.c2 = constraint.ConstraintsIntersection(self.c2, constraint.ValueRangeConstraint(1, 10))

    def testGoodVal(self):
        assert self.c1.isSuperTypeOf(self.c2), 'isSuperTypeOf failed'
        assert not self.c1.isSubTypeOf(self.c2), 'isSubTypeOf failed'

    def testBadVal(self):
        assert not self.c2.isSuperTypeOf(self.c1), 'isSuperTypeOf failed'
        assert self.c2.isSubTypeOf(self.c1), 'isSubTypeOf failed'