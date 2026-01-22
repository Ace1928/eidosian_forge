import re
import unittest
from oslo_config import types
class FloatTypeTests(TypeTestHelper, unittest.TestCase):
    type = types.Float()

    def test_decimal_format(self):
        v = self.type_instance('123.456')
        self.assertAlmostEqual(v, 123.456)

    def test_decimal_format_negative_float(self):
        v = self.type_instance('-123.456')
        self.assertAlmostEqual(v, -123.456)

    def test_exponential_format(self):
        v = self.type_instance('123e-2')
        self.assertAlmostEqual(v, 1.23)

    def test_non_float_is_invalid(self):
        self.assertInvalid('123,345')
        self.assertInvalid('foo')

    def test_repr(self):
        self.assertEqual('Float', repr(types.Float()))

    def test_repr_with_min(self):
        t = types.Float(min=1.1)
        self.assertEqual('Float(min=1.1)', repr(t))

    def test_repr_with_max(self):
        t = types.Float(max=2.2)
        self.assertEqual('Float(max=2.2)', repr(t))

    def test_repr_with_min_and_max(self):
        t = types.Float(min=1.1, max=2.2)
        self.assertEqual('Float(min=1.1, max=2.2)', repr(t))
        t = types.Float(min=1.0, max=2)
        self.assertEqual('Float(min=1, max=2)', repr(t))
        t = types.Float(min=0, max=0)
        self.assertEqual('Float(min=0, max=0)', repr(t))

    def test_equal(self):
        self.assertTrue(types.Float() == types.Float())

    def test_equal_with_same_min_and_no_max(self):
        self.assertTrue(types.Float(min=123.1) == types.Float(min=123.1))

    def test_equal_with_same_max_and_no_min(self):
        self.assertTrue(types.Float(max=123.1) == types.Float(max=123.1))

    def test_not_equal(self):
        self.assertFalse(types.Float(min=123.1) == types.Float(min=456.1))
        self.assertFalse(types.Float(max=123.1) == types.Float(max=456.1))
        self.assertFalse(types.Float(min=123.1) == types.Float(max=123.1))
        self.assertFalse(types.Float(min=123.1, max=456.1) == types.Float(min=123.1, max=456.2))

    def test_not_equal_to_other_class(self):
        self.assertFalse(types.Float() == types.Integer())

    def test_equal_with_same_min_and_max(self):
        t1 = types.Float(min=1.1, max=2.2)
        t2 = types.Float(min=1.1, max=2.2)
        self.assertTrue(t1 == t2)

    def test_min_greater_max(self):
        self.assertRaises(ValueError, types.Float, min=100.1, max=50)
        self.assertRaises(ValueError, types.Float, min=-50, max=-100.1)
        self.assertRaises(ValueError, types.Float, min=0.1, max=-50.0)
        self.assertRaises(ValueError, types.Float, min=50.0, max=0.0)

    def test_with_max_and_min(self):
        t = types.Float(min=123.45, max=678.9)
        self.assertRaises(ValueError, t, 123)
        self.assertRaises(ValueError, t, 123.1)
        t(124.1)
        t(300)
        t(456.0)
        self.assertRaises(ValueError, t, 0)
        self.assertRaises(ValueError, t, 800.5)

    def test_with_min_zero(self):
        t = types.Float(min=0, max=456.1)
        self.assertRaises(ValueError, t, -1)
        t(0.0)
        t(123.1)
        t(300.2)
        t(456.1)
        self.assertRaises(ValueError, t, -201.0)
        self.assertRaises(ValueError, t, 457.0)

    def test_with_max_zero(self):
        t = types.Float(min=-456.1, max=0)
        self.assertRaises(ValueError, t, 1)
        t(0.0)
        t(-123.0)
        t(-300.0)
        t(-456.0)
        self.assertRaises(ValueError, t, 201.0)
        self.assertRaises(ValueError, t, -457.0)