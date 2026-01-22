from unittest import mock
from oslo_serialization import jsonutils
from heat.common import exception
from heat.engine import constraints
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine.hot import parameters as hot_param
from heat.engine import parameters
from heat.engine import plugin_manager
from heat.engine import properties
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import support
from heat.engine import translation
from heat.tests import common
class PropertiesValidationTest(common.HeatTestCase):

    def test_required(self):
        schema = {'foo': {'Type': 'String', 'Required': True}}
        props = properties.Properties(schema, {'foo': 'bar'})
        self.assertIsNone(props.validate())

    def test_missing_required(self):
        schema = {'foo': {'Type': 'String', 'Required': True}}
        props = properties.Properties(schema, {})
        self.assertRaises(exception.StackValidationFailed, props.validate)

    def test_missing_unimplemented(self):
        schema = {'foo': {'Type': 'String', 'Implemented': False}}
        props = properties.Properties(schema, {})
        self.assertIsNone(props.validate())

    def test_present_unimplemented(self):
        schema = {'foo': {'Type': 'String', 'Implemented': False}}
        props = properties.Properties(schema, {'foo': 'bar'})
        self.assertRaises(exception.StackValidationFailed, props.validate)

    def test_missing(self):
        schema = {'foo': {'Type': 'String'}}
        props = properties.Properties(schema, {})
        self.assertIsNone(props.validate())

    def test_unknown_typo(self):
        schema = {'foo': {'Type': 'String'}}
        props = properties.Properties(schema, {'food': 42})
        self.assertRaises(exception.StackValidationFailed, props.validate)

    def test_list_instead_string(self):
        schema = {'foo': {'Type': 'String'}}
        props = properties.Properties(schema, {'foo': ['foo', 'bar']})
        ex = self.assertRaises(exception.StackValidationFailed, props.validate)
        self.assertIn('Property error: foo: Value must be a string', str(ex))

    def test_dict_instead_string(self):
        schema = {'foo': {'Type': 'String'}}
        props = properties.Properties(schema, {'foo': {'foo': 'bar'}})
        ex = self.assertRaises(exception.StackValidationFailed, props.validate)
        self.assertIn('Property error: foo: Value must be a string', str(ex))

    def test_none_string(self):
        schema = {'foo': {'Type': 'String'}}
        props = properties.Properties(schema, {'foo': None})
        self.assertIsNone(props.validate())

    def test_none_integer(self):
        schema = {'foo': {'Type': 'Integer'}}
        props = properties.Properties(schema, {'foo': None})
        self.assertIsNone(props.validate())

    def test_none_number(self):
        schema = {'foo': {'Type': 'Number'}}
        props = properties.Properties(schema, {'foo': None})
        self.assertIsNone(props.validate())

    def test_none_boolean(self):
        schema = {'foo': {'Type': 'Boolean'}}
        props = properties.Properties(schema, {'foo': None})
        self.assertIsNone(props.validate())

    def test_none_map(self):
        schema = {'foo': {'Type': 'Map'}}
        props = properties.Properties(schema, {'foo': None})
        self.assertIsNone(props.validate())

    def test_none_list(self):
        schema = {'foo': {'Type': 'List'}}
        props = properties.Properties(schema, {'foo': None})
        self.assertIsNone(props.validate())

    def test_none_default_string(self):
        schema = {'foo': {'Type': 'String', 'Default': 'bar'}}
        props = properties.Properties(schema, {'foo': None})
        self.assertIsNone(props.validate())

    def test_none_default_integer(self):
        schema = {'foo': {'Type': 'Integer', 'Default': 42}}
        props = properties.Properties(schema, {'foo': None})
        self.assertIsNone(props.validate())

    def test_none_default_number(self):
        schema = {'foo': {'Type': 'Number', 'Default': 42.0}}
        props = properties.Properties(schema, {'foo': None})
        self.assertIsNone(props.validate())

    def test_none_default_boolean(self):
        schema = {'foo': {'Type': 'Boolean', 'Default': True}}
        props = properties.Properties(schema, {'foo': None})
        self.assertIsNone(props.validate())

    def test_none_default_map(self):
        schema = {'foo': {'Type': 'Map', 'Default': {'bar': 'baz'}}}
        props = properties.Properties(schema, {'foo': None})
        self.assertIsNone(props.validate())

    def test_none_default_list(self):
        schema = {'foo': {'Type': 'List', 'Default': ['one', 'two']}}
        props = properties.Properties(schema, {'foo': None})
        self.assertIsNone(props.validate())

    def test_schema_to_template_nested_map_map_schema(self):
        nested_schema = {'Key': {'Type': 'String', 'Required': True}, 'Value': {'Type': 'String', 'Default': 'fewaf'}}
        schema = {'foo': {'Type': 'Map', 'Schema': nested_schema}}
        prop_expected = {'foo': {'Ref': 'foo'}}
        param_expected = {'foo': {'Type': 'Json'}}
        parameters, props = properties.Properties.schema_to_parameters_and_properties(schema)
        self.assertEqual(param_expected, parameters)
        self.assertEqual(prop_expected, props)

    def test_schema_to_template_nested_map_list_map_schema(self):
        key_schema = {'bar': {'Type': 'Number'}}
        nested_schema = {'Key': {'Type': 'Map', 'Schema': key_schema}, 'Value': {'Type': 'String', 'Required': True}}
        schema = {'foo': {'Type': 'List', 'Schema': {'Type': 'Map', 'Schema': nested_schema}}}
        prop_expected = {'foo': {'Fn::Split': [',', {'Ref': 'foo'}]}}
        param_expected = {'foo': {'Type': 'CommaDelimitedList'}}
        parameters, props = properties.Properties.schema_to_parameters_and_properties(schema)
        self.assertEqual(param_expected, parameters)
        self.assertEqual(prop_expected, props)

    def test_schema_object_to_template_nested_map_list_map_schema(self):
        key_schema = {'bar': properties.Schema(properties.Schema.NUMBER)}
        nested_schema = {'Key': properties.Schema(properties.Schema.MAP, schema=key_schema), 'Value': properties.Schema(properties.Schema.STRING, required=True)}
        schema = {'foo': properties.Schema(properties.Schema.LIST, schema=properties.Schema(properties.Schema.MAP, schema=nested_schema))}
        prop_expected = {'foo': {'Fn::Split': [',', {'Ref': 'foo'}]}}
        param_expected = {'foo': {'Type': 'CommaDelimitedList'}}
        parameters, props = properties.Properties.schema_to_parameters_and_properties(schema)
        self.assertEqual(param_expected, parameters)
        self.assertEqual(prop_expected, props)

    def test_schema_invalid_parameters_stripped(self):
        schema = {'foo': {'Type': 'String', 'Required': True, 'Implemented': True}}
        prop_expected = {'foo': {'Ref': 'foo'}}
        param_expected = {'foo': {'Type': 'String'}}
        parameters, props = properties.Properties.schema_to_parameters_and_properties(schema)
        self.assertEqual(param_expected, parameters)
        self.assertEqual(prop_expected, props)

    def test_schema_support_status(self):
        schema = {'foo_sup': properties.Schema(properties.Schema.STRING, default='foo'), 'bar_dep': properties.Schema(properties.Schema.STRING, default='bar', support_status=support.SupportStatus(support.DEPRECATED, 'Do not use this ever'))}
        props = properties.Properties(schema, {})
        self.assertEqual(support.SUPPORTED, props.props['foo_sup'].support_status().status)
        self.assertEqual(support.DEPRECATED, props.props['bar_dep'].support_status().status)
        self.assertEqual('Do not use this ever', props.props['bar_dep'].support_status().message)

    def test_nested_properties_schema_invalid_property_in_list(self):
        child_schema = {'Key': {'Type': 'String', 'Required': True}, 'Value': {'Type': 'Boolean', 'Default': True}}
        list_schema = {'Type': 'Map', 'Schema': child_schema}
        schema = {'foo': {'Type': 'List', 'Schema': list_schema}}
        valid_data = {'foo': [{'Key': 'Test'}]}
        props = properties.Properties(schema, valid_data)
        self.assertIsNone(props.validate())
        invalid_data = {'foo': [{'Key': 'Test', 'bar': 'baz'}]}
        props = properties.Properties(schema, invalid_data)
        ex = self.assertRaises(exception.StackValidationFailed, props.validate)
        self.assertEqual('Property error: foo[0]: Unknown Property bar', str(ex))

    def test_nested_properties_schema_invalid_property_in_map(self):
        child_schema = {'Key': {'Type': 'String', 'Required': True}, 'Value': {'Type': 'Boolean', 'Default': True}}
        map_schema = {'boo': {'Type': 'Map', 'Schema': child_schema}}
        schema = {'foo': {'Type': 'Map', 'Schema': map_schema}}
        valid_data = {'foo': {'boo': {'Key': 'Test'}}}
        props = properties.Properties(schema, valid_data)
        self.assertIsNone(props.validate())
        invalid_data = {'foo': {'boo': {'Key': 'Test', 'bar': 'baz'}}}
        props = properties.Properties(schema, invalid_data)
        ex = self.assertRaises(exception.StackValidationFailed, props.validate)
        self.assertEqual('Property error: foo.boo: Unknown Property bar', str(ex))

    def test_more_nested_properties_schema_invalid_property_in_list(self):
        nested_child_schema = {'Key': {'Type': 'String', 'Required': True}}
        child_schema = {'doo': {'Type': 'Map', 'Schema': nested_child_schema}}
        list_schema = {'Type': 'Map', 'Schema': child_schema}
        schema = {'foo': {'Type': 'List', 'Schema': list_schema}}
        valid_data = {'foo': [{'doo': {'Key': 'Test'}}]}
        props = properties.Properties(schema, valid_data)
        self.assertIsNone(props.validate())
        invalid_data = {'foo': [{'doo': {'Key': 'Test', 'bar': 'baz'}}]}
        props = properties.Properties(schema, invalid_data)
        ex = self.assertRaises(exception.StackValidationFailed, props.validate)
        self.assertEqual('Property error: foo[0].doo: Unknown Property bar', str(ex))

    def test_more_nested_properties_schema_invalid_property_in_map(self):
        nested_child_schema = {'Key': {'Type': 'String', 'Required': True}}
        child_schema = {'doo': {'Type': 'Map', 'Schema': nested_child_schema}}
        map_schema = {'boo': {'Type': 'Map', 'Schema': child_schema}}
        schema = {'foo': {'Type': 'Map', 'Schema': map_schema}}
        valid_data = {'foo': {'boo': {'doo': {'Key': 'Test'}}}}
        props = properties.Properties(schema, valid_data)
        self.assertIsNone(props.validate())
        invalid_data = {'foo': {'boo': {'doo': {'Key': 'Test', 'bar': 'baz'}}}}
        props = properties.Properties(schema, invalid_data)
        ex = self.assertRaises(exception.StackValidationFailed, props.validate)
        self.assertEqual('Property error: foo.boo.doo: Unknown Property bar', str(ex))

    def test_schema_to_template_empty_schema(self):
        schema = {}
        parameters, props = properties.Properties.schema_to_parameters_and_properties(schema)
        self.assertEqual({}, parameters)
        self.assertEqual({}, props)

    def test_update_allowed_and_immutable_contradict(self):
        self.assertRaises(exception.InvalidSchemaError, properties.Schema, properties.Schema.STRING, update_allowed=True, immutable=True)