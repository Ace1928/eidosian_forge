from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
class ParameterTestSpecific(common.HeatTestCase):

    def test_new_bad_type(self):
        self.assertRaises(exception.InvalidSchemaError, new_parameter, 'p', {'Type': 'List'}, validate_value=False)

    def test_string_len_good(self):
        schema = {'Type': 'String', 'MinLength': '3', 'MaxLength': '3'}
        p = new_parameter('p', schema, 'foo')
        self.assertEqual('foo', p.value())

    def test_string_underflow(self):
        schema = {'Type': 'String', 'ConstraintDescription': 'wibble', 'MinLength': '4'}
        err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'p', schema, 'foo')
        self.assertIn('wibble', str(err))

    def test_string_overflow(self):
        schema = {'Type': 'String', 'ConstraintDescription': 'wibble', 'MaxLength': '2'}
        err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'p', schema, 'foo')
        self.assertIn('wibble', str(err))

    def test_string_pattern_good(self):
        schema = {'Type': 'String', 'AllowedPattern': '[a-z]*'}
        p = new_parameter('p', schema, 'foo')
        self.assertEqual('foo', p.value())

    def test_string_pattern_bad_prefix(self):
        schema = {'Type': 'String', 'ConstraintDescription': 'wibble', 'AllowedPattern': '[a-z]*'}
        err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'p', schema, '1foo')
        self.assertIn('wibble', str(err))

    def test_string_pattern_bad_suffix(self):
        schema = {'Type': 'String', 'ConstraintDescription': 'wibble', 'AllowedPattern': '[a-z]*'}
        err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'p', schema, 'foo1')
        self.assertIn('wibble', str(err))

    def test_string_value_list_good(self):
        schema = {'Type': 'String', 'AllowedValues': ['foo', 'bar', 'baz']}
        p = new_parameter('p', schema, 'bar')
        self.assertEqual('bar', p.value())

    def test_string_value_unicode(self):
        schema = {'Type': 'String'}
        p = new_parameter('p', schema, u'test♥')
        self.assertEqual(u'test♥', p.value())

    def test_string_value_list_bad(self):
        schema = {'Type': 'String', 'ConstraintDescription': 'wibble', 'AllowedValues': ['foo', 'bar', 'baz']}
        err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'p', schema, 'blarg')
        self.assertIn('wibble', str(err))

    def test_number_int_good(self):
        schema = {'Type': 'Number', 'MinValue': '3', 'MaxValue': '3'}
        p = new_parameter('p', schema, '3')
        self.assertEqual(3, p.value())

    def test_number_float_good_string(self):
        schema = {'Type': 'Number', 'MinValue': '3.0', 'MaxValue': '4.0'}
        p = new_parameter('p', schema, '3.5')
        self.assertEqual(3.5, p.value())

    def test_number_float_good_number(self):
        schema = {'Type': 'Number', 'MinValue': '3.0', 'MaxValue': '4.0'}
        p = new_parameter('p', schema, 3.5)
        self.assertEqual(3.5, p.value())

    def test_number_low(self):
        schema = {'Type': 'Number', 'ConstraintDescription': 'wibble', 'MinValue': '4'}
        err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'p', schema, '3')
        self.assertIn('wibble', str(err))

    def test_number_high(self):
        schema = {'Type': 'Number', 'ConstraintDescription': 'wibble', 'MaxValue': '2'}
        err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'p', schema, '3')
        self.assertIn('wibble', str(err))

    def test_number_bad(self):
        schema = {'Type': 'Number'}
        err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'p', schema, 'str')
        self.assertIn('float', str(err))

    def test_number_bad_type(self):
        schema = {'Type': 'Number'}
        err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'p', schema, ['foo'])
        self.assertIn('int', str(err))

    def test_number_value_list_good(self):
        schema = {'Type': 'Number', 'AllowedValues': ['1', '3', '5']}
        p = new_parameter('p', schema, '5')
        self.assertEqual(5, p.value())

    def test_number_value_list_bad(self):
        schema = {'Type': 'Number', 'ConstraintDescription': 'wibble', 'AllowedValues': ['1', '3', '5']}
        err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'p', schema, '2')
        self.assertIn('wibble', str(err))

    def test_list_value_list_default_empty(self):
        schema = {'Type': 'CommaDelimitedList', 'Default': ''}
        p = new_parameter('p', schema)
        self.assertEqual([], p.value())

    def test_list_value_list_good(self):
        schema = {'Type': 'CommaDelimitedList', 'AllowedValues': ['foo', 'bar', 'baz']}
        p = new_parameter('p', schema, 'baz,foo,bar')
        self.assertEqual('baz,foo,bar'.split(','), p.value())
        schema['Default'] = []
        p = new_parameter('p', schema)
        self.assertEqual([], p.value())
        schema['Default'] = 'baz,foo,bar'
        p = new_parameter('p', schema)
        self.assertEqual('baz,foo,bar'.split(','), p.value())
        schema['AllowedValues'] = ['1', '3', '5']
        schema['Default'] = []
        p = new_parameter('p', schema, [1, 3, 5])
        self.assertEqual('1,3,5', str(p))
        schema['Default'] = [1, 3, 5]
        p = new_parameter('p', schema)
        self.assertEqual('1,3,5'.split(','), p.value())

    def test_list_value_list_bad(self):
        schema = {'Type': 'CommaDelimitedList', 'ConstraintDescription': 'wibble', 'AllowedValues': ['foo', 'bar', 'baz']}
        err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'p', schema, 'foo,baz,blarg')
        self.assertIn('wibble', str(err))

    def test_list_validate_good(self):
        schema = {'Type': 'CommaDelimitedList'}
        val = ['foo', 'bar', 'baz']
        val_s = 'foo,bar,baz'
        p = new_parameter('p', schema, val_s, validate_value=False)
        p.validate()
        self.assertEqual(val, p.value())
        self.assertEqual(val, p.parsed)

    def test_list_validate_bad(self):
        schema = {'Type': 'CommaDelimitedList'}
        val_s = 0
        p = new_parameter('p', schema, validate_value=False)
        p.user_value = val_s
        err = self.assertRaises(exception.StackValidationFailed, p.validate)
        self.assertIn("Parameter 'p' is invalid", str(err))

    def test_map_value(self):
        """Happy path for value that's already a map."""
        schema = {'Type': 'Json'}
        val = {'foo': 'bar', 'items': [1, 2, 3]}
        p = new_parameter('p', schema, val)
        self.assertEqual(val, p.value())
        self.assertEqual(val, p.parsed)

    def test_map_value_bad(self):
        """Map value is not JSON parsable."""
        schema = {'Type': 'Json', 'ConstraintDescription': 'wibble'}
        val = {'foo': 'bar', 'not_json': len}
        err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'p', schema, val)
        self.assertIn('Value must be valid JSON', str(err))

    def test_map_value_parse(self):
        """Happy path for value that's a string."""
        schema = {'Type': 'Json'}
        val = {'foo': 'bar', 'items': [1, 2, 3]}
        val_s = json.dumps(val)
        p = new_parameter('p', schema, val_s)
        self.assertEqual(val, p.value())
        self.assertEqual(val, p.parsed)

    def test_map_value_bad_parse(self):
        """Test value error for unparsable string value."""
        schema = {'Type': 'Json', 'ConstraintDescription': 'wibble'}
        val = 'I am not a map'
        err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'p', schema, val)
        self.assertIn('Value must be valid JSON', str(err))

    def test_map_underrun(self):
        """Test map length under MIN_LEN."""
        schema = {'Type': 'Json', 'MinLength': 3}
        val = {'foo': 'bar', 'items': [1, 2, 3]}
        err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'p', schema, val)
        self.assertIn('out of range', str(err))

    def test_map_overrun(self):
        """Test map length over MAX_LEN."""
        schema = {'Type': 'Json', 'MaxLength': 1}
        val = {'foo': 'bar', 'items': [1, 2, 3]}
        err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'p', schema, val)
        self.assertIn('out of range', str(err))

    def test_json_list(self):
        schema = {'Type': 'Json'}
        val = ['fizz', 'buzz']
        p = new_parameter('p', schema, val)
        self.assertIsInstance(p.value(), list)
        self.assertIn('fizz', p.value())
        self.assertIn('buzz', p.value())

    def test_json_string_list(self):
        schema = {'Type': 'Json'}
        val = '["fizz", "buzz"]'
        p = new_parameter('p', schema, val)
        self.assertIsInstance(p.value(), list)
        self.assertIn('fizz', p.value())
        self.assertIn('buzz', p.value())

    def test_json_validate_good(self):
        schema = {'Type': 'Json'}
        val = {'foo': 'bar', 'items': [1, 2, 3]}
        val_s = json.dumps(val)
        p = new_parameter('p', schema, val_s, validate_value=False)
        p.validate()
        self.assertEqual(val, p.value())
        self.assertEqual(val, p.parsed)

    def test_json_validate_bad(self):
        schema = {'Type': 'Json'}
        val_s = '{"foo": "bar", "invalid": ]}'
        p = new_parameter('p', schema, validate_value=False)
        p.user_value = val_s
        err = self.assertRaises(exception.StackValidationFailed, p.validate)
        self.assertIn("Parameter 'p' is invalid", str(err))

    def test_bool_value_true(self):
        schema = {'Type': 'Boolean'}
        for val in ('1', 't', 'true', 'on', 'y', 'yes', True, 1):
            bo = new_parameter('bo', schema, val)
            self.assertTrue(bo.value())

    def test_bool_value_false(self):
        schema = {'Type': 'Boolean'}
        for val in ('0', 'f', 'false', 'off', 'n', 'no', False, 0):
            bo = new_parameter('bo', schema, val)
            self.assertFalse(bo.value())

    def test_bool_value_invalid(self):
        schema = {'Type': 'Boolean'}
        err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'bo', schema, 'foo')
        self.assertIn("Unrecognized value 'foo'", str(err))

    def test_missing_param_str(self):
        """Test missing user parameter."""
        self.assertRaises(exception.UserParameterMissing, new_parameter, 'p', {'Type': 'String'})

    def test_missing_param_list(self):
        """Test missing user parameter."""
        self.assertRaises(exception.UserParameterMissing, new_parameter, 'p', {'Type': 'CommaDelimitedList'})

    def test_missing_param_map(self):
        """Test missing user parameter."""
        self.assertRaises(exception.UserParameterMissing, new_parameter, 'p', {'Type': 'Json'})

    def test_param_name_in_error_message(self):
        schema = {'Type': 'String', 'AllowedPattern': '[a-z]*'}
        err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'testparam', schema, '234')
        expected = 'Parameter \'testparam\' is invalid: "234" does not match pattern "[a-z]*"'
        self.assertEqual(expected, str(err))