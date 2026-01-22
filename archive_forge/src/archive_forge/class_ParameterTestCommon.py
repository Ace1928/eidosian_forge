from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
class ParameterTestCommon(common.HeatTestCase):
    scenarios = [('type_string', dict(p_type='String', inst=parameters.StringParam, value='test', expected='test', allowed_value=['foo'], zero='', default='default')), ('type_number', dict(p_type='Number', inst=parameters.NumberParam, value=10, expected='10', allowed_value=[42], zero=0, default=13)), ('type_list', dict(p_type='CommaDelimitedList', inst=parameters.CommaDelimitedListParam, value=['a', 'b', 'c'], expected='a,b,c', allowed_value=['foo'], zero=[], default=['d', 'e', 'f'])), ('type_json', dict(p_type='Json', inst=parameters.JsonParam, value={'a': '1'}, expected='{"a": "1"}', allowed_value=[{'foo': 'bar'}], zero={}, default={'d': '1'})), ('type_int_json', dict(p_type='Json', inst=parameters.JsonParam, value={'a': 1}, expected='{"a": 1}', allowed_value=[{'foo': 'bar'}], zero={}, default={'d': 1})), ('type_boolean', dict(p_type='Boolean', inst=parameters.BooleanParam, value=True, expected='True', allowed_value=[False], zero=False, default=True)), ('type_int_string', dict(p_type='String', inst=parameters.StringParam, value='111', expected='111', allowed_value=['111'], zero='', default='0')), ('type_string_json', dict(p_type='Json', inst=parameters.JsonParam, value={'1': 1}, expected='{"1": 1}', allowed_value=[{'2': '2'}], zero={}, default={'3': 3}))]

    def test_new_param(self):
        p = new_parameter('p', {'Type': self.p_type}, validate_value=False)
        self.assertIsInstance(p, self.inst)

    def test_param_to_str(self):
        p = new_parameter('p', {'Type': self.p_type}, self.value)
        if self.p_type == 'Json':
            self.assertEqual(json.loads(self.expected), json.loads(str(p)))
        else:
            self.assertEqual(self.expected, str(p))

    def test_default_no_override(self):
        p = new_parameter('defaulted', {'Type': self.p_type, 'Default': self.default})
        self.assertTrue(p.has_default())
        self.assertEqual(self.default, p.default())
        self.assertEqual(self.default, p.value())

    def test_default_override(self):
        p = new_parameter('defaulted', {'Type': self.p_type, 'Default': self.default}, self.value)
        self.assertTrue(p.has_default())
        self.assertEqual(self.default, p.default())
        self.assertEqual(self.value, p.value())

    def test_default_invalid(self):
        schema = {'Type': self.p_type, 'AllowedValues': self.allowed_value, 'ConstraintDescription': 'wibble', 'Default': self.default}
        if self.p_type == 'Json':
            err = self.assertRaises(exception.InvalidSchemaError, new_parameter, 'p', schema)
            self.assertIn('AllowedValues constraint invalid for Json', str(err))
        else:
            err = self.assertRaises(exception.InvalidSchemaError, new_parameter, 'p', schema)
            self.assertIn('wibble', str(err))

    def test_description(self):
        description = 'Description of the parameter'
        p = new_parameter('p', {'Type': self.p_type, 'Description': description}, validate_value=False)
        self.assertEqual(description, p.description())

    def test_no_description(self):
        p = new_parameter('p', {'Type': self.p_type}, validate_value=False)
        self.assertEqual('', p.description())

    def test_no_echo_true(self):
        p = new_parameter('anechoic', {'Type': self.p_type, 'NoEcho': 'true'}, self.value)
        self.assertTrue(p.hidden())
        self.assertEqual('******', str(p))

    def test_no_echo_true_caps(self):
        p = new_parameter('anechoic', {'Type': self.p_type, 'NoEcho': 'TrUe'}, self.value)
        self.assertTrue(p.hidden())
        self.assertEqual('******', str(p))

    def test_no_echo_false(self):
        p = new_parameter('echoic', {'Type': self.p_type, 'NoEcho': 'false'}, self.value)
        self.assertFalse(p.hidden())
        if self.p_type == 'Json':
            self.assertEqual(json.loads(self.expected), json.loads(str(p)))
        else:
            self.assertEqual(self.expected, str(p))

    def test_default_empty(self):
        p = new_parameter('defaulted', {'Type': self.p_type, 'Default': self.zero})
        self.assertTrue(p.has_default())
        self.assertEqual(self.zero, p.default())
        self.assertEqual(self.zero, p.value())

    def test_default_no_empty_user_value_empty(self):
        p = new_parameter('defaulted', {'Type': self.p_type, 'Default': self.default}, self.zero)
        self.assertTrue(p.has_default())
        self.assertEqual(self.default, p.default())
        self.assertEqual(self.zero, p.value())