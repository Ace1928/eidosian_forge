import re
import unittest
from oslo_config import types
class StringTypeTests(TypeTestHelper, unittest.TestCase):
    type = types.String()

    def test_empty_string_passes(self):
        self.assertConvertedValue('', '')

    def test_should_return_same_string_if_valid(self):
        self.assertConvertedValue('foo bar', 'foo bar')

    def test_listed_value(self):
        self.type_instance = types.String(choices=['foo', 'bar'])
        self.assertConvertedValue('foo', 'foo')

    def test_listed_value_tuple(self):
        self.type_instance = types.String(choices=('foo', 'bar'))
        self.assertConvertedValue('foo', 'foo')

    def test_listed_value_dict(self):
        self.type_instance = types.String(choices=[('foo', 'ab'), ('bar', 'xy')])
        self.assertConvertedValue('foo', 'foo')

    def test_unlisted_value(self):
        self.type_instance = types.String(choices=['foo', 'bar'])
        self.assertInvalid('baz')

    def test_with_no_values_returns_error(self):
        self.type_instance = types.String(choices=[])
        self.assertInvalid('foo')

    def test_string_with_non_closed_quote_is_invalid(self):
        self.type_instance = types.String(quotes=True)
        self.assertInvalid('"foo bar')
        self.assertInvalid("'bar baz")

    def test_quotes_are_stripped(self):
        self.type_instance = types.String(quotes=True)
        self.assertConvertedValue('"foo bar"', 'foo bar')

    def test_trailing_quote_is_ok(self):
        self.type_instance = types.String(quotes=True)
        self.assertConvertedValue('foo bar"', 'foo bar"')

    def test_single_quote_is_invalid(self):
        self.type_instance = types.String(quotes=True)
        self.assertInvalid('"')
        self.assertInvalid("'")

    def test_repr(self):
        t = types.String()
        self.assertEqual('String', repr(t))

    def test_repr_with_choices(self):
        t = types.String(choices=['foo', 'bar'])
        self.assertEqual("String(choices=['foo', 'bar'])", repr(t))

    def test_repr_with_choices_tuple(self):
        t = types.String(choices=('foo', 'bar'))
        self.assertEqual("String(choices=['foo', 'bar'])", repr(t))

    def test_repr_with_choices_dict(self):
        t = types.String(choices=[('foo', 'ab'), ('bar', 'xy')])
        self.assertEqual("String(choices=['foo', 'bar'])", repr(t))

    def test_equal(self):
        self.assertTrue(types.String() == types.String())

    def test_equal_with_same_choices(self):
        t1 = types.String(choices=['foo', 'bar'])
        t2 = types.String(choices=['foo', 'bar'])
        t3 = types.String(choices=('foo', 'bar'))
        t4 = types.String(choices=['bar', 'foo'])
        t5 = types.String(choices=[('foo', 'ab'), ('bar', 'xy')])
        self.assertTrue(t1 == t2 == t3 == t4 == t5)

    def test_not_equal_with_different_choices(self):
        t1 = types.String(choices=['foo', 'bar'])
        t2 = types.String(choices=['foo', 'baz'])
        t3 = types.String(choices=('foo', 'baz'))
        self.assertFalse(t1 == t2)
        self.assertFalse(t1 == t3)

    def test_equal_with_equal_quote_falgs(self):
        t1 = types.String(quotes=True)
        t2 = types.String(quotes=True)
        self.assertTrue(t1 == t2)

    def test_not_equal_with_different_quote_falgs(self):
        t1 = types.String(quotes=False)
        t2 = types.String(quotes=True)
        self.assertFalse(t1 == t2)

    def test_not_equal_to_other_class(self):
        self.assertFalse(types.String() == types.Integer())

    def test_regex_matches(self):
        self.type_instance = types.String(regex=re.compile('^[A-Z]'))
        self.assertConvertedValue('Foo', 'Foo')

    def test_regex_matches_uncompiled(self):
        self.type_instance = types.String(regex='^[A-Z]')
        self.assertConvertedValue('Foo', 'Foo')

    def test_regex_fails(self):
        self.type_instance = types.String(regex=re.compile('^[A-Z]'))
        self.assertInvalid('foo')

    def test_regex_and_choices_raises(self):
        self.assertRaises(ValueError, types.String, regex=re.compile('^[A-Z]'), choices=['Foo', 'Bar', 'baz'])

    def test_equal_with_same_regex(self):
        t1 = types.String(regex=re.compile('^[A-Z]'))
        t2 = types.String(regex=re.compile('^[A-Z]'))
        self.assertTrue(t1 == t2)

    def test_not_equal_with_different_regex(self):
        t1 = types.String(regex=re.compile('^[A-Z]'))
        t2 = types.String(regex=re.compile('^[a-z]'))
        self.assertFalse(t1 == t2)

    def test_ignore_case(self):
        self.type_instance = types.String(choices=['foo', 'bar'], ignore_case=True)
        self.assertConvertedValue('Foo', 'Foo')
        self.assertConvertedValue('bAr', 'bAr')

    def test_ignore_case_raises(self):
        self.type_instance = types.String(choices=['foo', 'bar'], ignore_case=False)
        self.assertRaises(ValueError, self.assertConvertedValue, 'Foo', 'Foo')

    def test_regex_and_ignore_case(self):
        self.type_instance = types.String(regex=re.compile('^[A-Z]'), ignore_case=True)
        self.assertConvertedValue('foo', 'foo')

    def test_regex_and_ignore_case_str(self):
        self.type_instance = types.String(regex='^[A-Z]', ignore_case=True)
        self.assertConvertedValue('foo', 'foo')

    def test_regex_preserve_flags(self):
        self.type_instance = types.String(regex=re.compile('^[A-Z]', re.I), ignore_case=False)
        self.assertConvertedValue('foo', 'foo')

    def test_max_length(self):
        self.type_instance = types.String(max_length=5)
        self.assertInvalid('123456')
        self.assertConvertedValue('12345', '12345')