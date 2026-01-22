import re
import unittest
from oslo_config import types
class IntegerTypeTests(TypeTestHelper, unittest.TestCase):
    type = types.Integer()

    def test_empty_string(self):
        self.assertConvertedValue('', None)

    def test_whitespace_string(self):
        self.assertConvertedValue('   \t\t\t\t', None)

    def test_positive_values_are_valid(self):
        self.assertConvertedValue('123', 123)

    def test_zero_is_valid(self):
        self.assertConvertedValue('0', 0)

    def test_negative_values_are_valid(self):
        self.assertConvertedValue('-123', -123)

    def test_leading_whitespace_is_ignored(self):
        self.assertConvertedValue('   5', 5)

    def test_trailing_whitespace_is_ignored(self):
        self.assertConvertedValue('7   ', 7)

    def test_non_digits_are_invalid(self):
        self.assertInvalid('12a45')

    def test_repr(self):
        t = types.Integer()
        self.assertEqual('Integer', repr(t))

    def test_repr_with_min(self):
        t = types.Integer(min=123)
        self.assertEqual('Integer(min=123)', repr(t))

    def test_repr_with_max(self):
        t = types.Integer(max=456)
        self.assertEqual('Integer(max=456)', repr(t))

    def test_repr_with_min_and_max(self):
        t = types.Integer(min=123, max=456)
        self.assertEqual('Integer(min=123, max=456)', repr(t))
        t = types.Integer(min=0, max=0)
        self.assertEqual('Integer(min=0, max=0)', repr(t))

    def test_repr_with_choices(self):
        t = types.Integer(choices=[80, 457])
        self.assertEqual('Integer(choices=[80, 457])', repr(t))

    def test_repr_with_choices_tuple(self):
        t = types.Integer(choices=(80, 457))
        self.assertEqual('Integer(choices=[80, 457])', repr(t))

    def test_repr_with_choices_dict(self):
        t = types.Integer(choices=[(80, 'ab'), (457, 'xy')])
        self.assertEqual('Integer(choices=[80, 457])', repr(t))

    def test_equal(self):
        self.assertTrue(types.Integer() == types.Integer())

    def test_equal_with_same_min_and_no_max(self):
        self.assertTrue(types.Integer(min=123) == types.Integer(min=123))

    def test_equal_with_same_max_and_no_min(self):
        self.assertTrue(types.Integer(max=123) == types.Integer(max=123))

    def test_equal_with_same_min_and_max(self):
        t1 = types.Integer(min=1, max=123)
        t2 = types.Integer(min=1, max=123)
        self.assertTrue(t1 == t2)

    def test_equal_with_same_choices(self):
        t1 = types.Integer(choices=[80, 457])
        t2 = types.Integer(choices=[457, 80])
        t3 = types.Integer(choices=(457, 80))
        t4 = types.Integer(choices=[(80, 'ab'), (457, 'xy')])
        self.assertTrue(t1 == t2 == t3 == t4)

    def test_not_equal(self):
        self.assertFalse(types.Integer(min=123) == types.Integer(min=456))
        self.assertFalse(types.Integer(choices=[80, 457]) == types.Integer(choices=[80, 40]))
        self.assertFalse(types.Integer(choices=[80, 457]) == types.Integer())

    def test_not_equal_to_other_class(self):
        self.assertFalse(types.Integer() == types.String())

    def test_choices_with_min_max(self):
        self.assertRaises(ValueError, types.Integer, min=100, choices=[50, 60])
        self.assertRaises(ValueError, types.Integer, max=10, choices=[50, 60])
        types.Integer(min=10, max=100, choices=[50, 60])

    def test_min_greater_max(self):
        self.assertRaises(ValueError, types.Integer, min=100, max=50)
        self.assertRaises(ValueError, types.Integer, min=-50, max=-100)
        self.assertRaises(ValueError, types.Integer, min=0, max=-50)
        self.assertRaises(ValueError, types.Integer, min=50, max=0)

    def test_with_max_and_min(self):
        t = types.Integer(min=123, max=456)
        self.assertRaises(ValueError, t, 122)
        t(123)
        t(300)
        t(456)
        self.assertRaises(ValueError, t, 0)
        self.assertRaises(ValueError, t, 457)

    def test_with_min_zero(self):
        t = types.Integer(min=0, max=456)
        self.assertRaises(ValueError, t, -1)
        t(0)
        t(123)
        t(300)
        t(456)
        self.assertRaises(ValueError, t, -201)
        self.assertRaises(ValueError, t, 457)

    def test_with_max_zero(self):
        t = types.Integer(min=-456, max=0)
        self.assertRaises(ValueError, t, 1)
        t(0)
        t(-123)
        t(-300)
        t(-456)
        self.assertRaises(ValueError, t, 201)
        self.assertRaises(ValueError, t, -457)

    def _test_with_choices(self, t):
        self.assertRaises(ValueError, t, 1)
        self.assertRaises(ValueError, t, 200)
        self.assertRaises(ValueError, t, -457)
        t(80)
        t(457)

    def test_with_choices_list(self):
        t = types.Integer(choices=[80, 457])
        self._test_with_choices(t)

    def test_with_choices_tuple(self):
        t = types.Integer(choices=(80, 457))
        self._test_with_choices(t)

    def test_with_choices_dict(self):
        t = types.Integer(choices=[(80, 'ab'), (457, 'xy')])
        self._test_with_choices(t)