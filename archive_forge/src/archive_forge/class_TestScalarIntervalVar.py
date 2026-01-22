import pyomo.common.unittest as unittest
from pyomo.contrib.cp.interval_var import (
from pyomo.core.expr import GetItemExpression, GetAttrExpression
from pyomo.environ import ConcreteModel, Integers, Set, value, Var
class TestScalarIntervalVar(unittest.TestCase):

    def test_initialize_with_no_data(self):
        m = ConcreteModel()
        m.i = IntervalVar()
        self.assertIsInstance(m.i.start_time, IntervalVarTimePoint)
        self.assertEqual(m.i.start_time.domain, Integers)
        self.assertIsNone(m.i.start_time.lower)
        self.assertIsNone(m.i.start_time.upper)
        self.assertIsInstance(m.i.end_time, IntervalVarTimePoint)
        self.assertEqual(m.i.end_time.domain, Integers)
        self.assertIsNone(m.i.end_time.lower)
        self.assertIsNone(m.i.end_time.upper)
        self.assertIsInstance(m.i.length, IntervalVarLength)
        self.assertEqual(m.i.length.domain, Integers)
        self.assertIsNone(m.i.length.lower)
        self.assertIsNone(m.i.length.upper)
        self.assertIsInstance(m.i.is_present, IntervalVarPresence)

    def test_add_components_that_do_not_belong(self):
        m = ConcreteModel()
        m.i = IntervalVar()
        with self.assertRaisesRegex(ValueError, 'Attempting to declare a block component using the name of a reserved attribute:\n\tnew_thing'):
            m.i.new_thing = IntervalVar()

    def test_start_and_end_bounds(self):
        m = ConcreteModel()
        m.i = IntervalVar(start=(0, 5))
        self.assertEqual(m.i.start_time.lower, 0)
        self.assertEqual(m.i.start_time.upper, 5)
        m.i.end_time.bounds = (12, 14)
        self.assertEqual(m.i.end_time.lower, 12)
        self.assertEqual(m.i.end_time.upper, 14)

    def test_constant_length_and_start(self):
        m = ConcreteModel()
        m.i = IntervalVar(length=7, start=3)
        self.assertEqual(m.i.length.lower, 7)
        self.assertEqual(m.i.length.upper, 7)
        self.assertEqual(m.i.start_time.lower, 3)
        self.assertEqual(m.i.start_time.upper, 3)

    def test_non_optional(self):
        m = ConcreteModel()
        m.i = IntervalVar(length=2, end=(4, 9), optional=False)
        self.assertEqual(value(m.i.is_present), True)
        self.assertTrue(m.i.is_present.fixed)
        self.assertFalse(m.i.optional)
        m.i2 = IntervalVar()
        self.assertEqual(value(m.i2.is_present), True)
        self.assertTrue(m.i.is_present.fixed)
        self.assertFalse(m.i2.optional)

    def test_optional(self):
        m = ConcreteModel()
        m.i = IntervalVar(optional=True)
        self.assertFalse(m.i.is_present.fixed)
        self.assertTrue(m.i.optional)
        m.i.optional = False
        self.assertEqual(value(m.i.is_present), True)
        self.assertTrue(m.i.is_present.fixed)
        self.assertFalse(m.i.optional)

    def test_is_present_fixed_False(self):
        m = ConcreteModel()
        m.i = IntervalVar(optional=True)
        m.i.is_present.fix(False)
        self.assertTrue(m.i.optional)