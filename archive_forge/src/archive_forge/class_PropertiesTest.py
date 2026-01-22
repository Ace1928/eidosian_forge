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
class PropertiesTest(common.HeatTestCase):

    def setUp(self):
        super(PropertiesTest, self).setUp()
        schema = {'int': {'Type': 'Integer'}, 'string': {'Type': 'String'}, 'required_int': {'Type': 'Integer', 'Required': True}, 'bad_int': {'Type': 'Integer'}, 'missing': {'Type': 'Integer'}, 'defaulted': {'Type': 'Integer', 'Default': 1}, 'default_override': {'Type': 'Integer', 'Default': 1}, 'default_bool': {'Type': 'Boolean', 'Default': 'false'}}
        data = {'int': 21, 'string': 'foo', 'bad_int': 'foo', 'default_override': 21}

        def double(d, nullable=False):
            return d * 2
        self.props = properties.Properties(schema, data, double, 'wibble')

    def test_integer_good(self):
        self.assertEqual(42, self.props['int'])

    def test_string_good(self):
        self.assertEqual('foofoo', self.props['string'])

    def test_bool_not_str(self):
        self.assertFalse(self.props['default_bool'])

    def test_missing_required(self):
        self.assertRaises(ValueError, self.props.get, 'required_int')

    @mock.patch.object(translation.Translation, 'has_translation')
    @mock.patch.object(translation.Translation, 'translate')
    def test_required_with_translate_no_value(self, m_t, m_ht):
        m_t.return_value = None
        m_ht.return_value = True
        self.assertRaises(ValueError, self.props.get, 'required_int')

    def test_integer_bad(self):
        self.assertRaises(ValueError, self.props.get, 'bad_int')

    def test_missing(self):
        self.assertIsNone(self.props['missing'])

    def test_default(self):
        self.assertEqual(1, self.props['defaulted'])

    def test_default_override(self):
        self.assertEqual(42, self.props['default_override'])

    def test_get_user_value(self):
        self.assertIsNone(self.props.get_user_value('defaulted'))
        self.assertEqual(42, self.props.get_user_value('default_override'))

    def test_get_user_value_key_error(self):
        ex = self.assertRaises(KeyError, self.props.get_user_value, 'foo')
        self.assertEqual('Invalid Property foo', str(ex.args[0]))

    def test_bad_key(self):
        self.assertEqual('wibble', self.props.get('foo', 'wibble'))

    def test_key_error(self):
        ex = self.assertRaises(KeyError, self.props.__getitem__, 'foo')
        self.assertEqual('Invalid Property foo', str(ex.args[0]))

    def test_none_string(self):
        schema = {'foo': {'Type': 'String'}}
        props = properties.Properties(schema, {'foo': None})
        self.assertEqual('', props['foo'])

    def test_none_integer(self):
        schema = {'foo': {'Type': 'Integer'}}
        props = properties.Properties(schema, {'foo': None})
        self.assertEqual(0, props['foo'])

    def test_none_number(self):
        schema = {'foo': {'Type': 'Number'}}
        props = properties.Properties(schema, {'foo': None})
        self.assertEqual(0, props['foo'])

    def test_none_boolean(self):
        schema = {'foo': {'Type': 'Boolean'}}
        props = properties.Properties(schema, {'foo': None})
        self.assertIs(False, props['foo'])

    def test_none_map(self):
        schema = {'foo': {'Type': 'Map'}}
        props = properties.Properties(schema, {'foo': None})
        self.assertEqual({}, props['foo'])

    def test_none_list(self):
        schema = {'foo': {'Type': 'List'}}
        props = properties.Properties(schema, {'foo': None})
        self.assertEqual([], props['foo'])

    def test_none_default_string(self):
        schema = {'foo': {'Type': 'String', 'Default': 'bar'}}
        props = properties.Properties(schema, {'foo': None})
        self.assertEqual('bar', props['foo'])

    def test_none_default_integer(self):
        schema = {'foo': {'Type': 'Integer', 'Default': 42}}
        props = properties.Properties(schema, {'foo': None})
        self.assertEqual(42, props['foo'])
        schema = {'foo': {'Type': 'Integer', 'Default': 0}}
        props = properties.Properties(schema, {'foo': None})
        self.assertEqual(0, props['foo'])
        schema = {'foo': {'Type': 'Integer', 'Default': -273}}
        props = properties.Properties(schema, {'foo': None})
        self.assertEqual(-273, props['foo'])

    def test_none_default_number(self):
        schema = {'foo': {'Type': 'Number', 'Default': 42.0}}
        props = properties.Properties(schema, {'foo': None})
        self.assertEqual(42.0, props['foo'])
        schema = {'foo': {'Type': 'Number', 'Default': 0.0}}
        props = properties.Properties(schema, {'foo': None})
        self.assertEqual(0.0, props['foo'])
        schema = {'foo': {'Type': 'Number', 'Default': -273.15}}
        props = properties.Properties(schema, {'foo': None})
        self.assertEqual(-273.15, props['foo'])

    def test_none_default_boolean(self):
        schema = {'foo': {'Type': 'Boolean', 'Default': True}}
        props = properties.Properties(schema, {'foo': None})
        self.assertIs(True, props['foo'])

    def test_none_default_map(self):
        schema = {'foo': {'Type': 'Map', 'Default': {'bar': 'baz'}}}
        props = properties.Properties(schema, {'foo': None})
        self.assertEqual({'bar': 'baz'}, props['foo'])

    def test_none_default_list(self):
        schema = {'foo': {'Type': 'List', 'Default': ['one', 'two']}}
        props = properties.Properties(schema, {'foo': None})
        self.assertEqual(['one', 'two'], props['foo'])

    def test_resolve_returns_none(self):
        schema = {'foo': {'Type': 'String', 'MinLength': '5'}}

        def test_resolver(prop, nullable=False):
            return None
        self.patchobject(properties.Properties, '_find_deps_any_in_init').return_value = True
        props = properties.Properties(schema, {'foo': 'get_attr: [db, value]'}, test_resolver)
        try:
            self.assertIsNone(props.validate())
        except exception.StackValidationFailed:
            self.fail('Constraints should not have been evaluated.')

    def test_resolve_ref_with_constraints(self):

        class IncorrectConstraint(constraints.BaseCustomConstraint):
            expected_exceptions = (Exception,)

            def validate_with_client(self, client, value):
                raise Exception('Test exception')

        class TestCustomConstraint(constraints.CustomConstraint):

            @property
            def custom_constraint(self):
                return IncorrectConstraint()
        schema = {'foo': properties.Schema(properties.Schema.STRING, constraints=[TestCustomConstraint('test_constraint')])}

        def test_resolver(prop, nullable=False):
            return 'None'

        class rsrc(object):
            action = INIT = 'INIT'

        class DummyStack(dict):
            pass
        stack = DummyStack(another_res=rsrc())
        props = properties.Properties(schema, {'foo': hot_funcs.GetResource(stack, 'get_resource', 'another_res')}, test_resolver)
        try:
            self.assertIsNone(props.validate())
        except exception.StackValidationFailed:
            self.fail('Constraints should not have been evaluated.')

    def test_schema_from_params(self):
        params_snippet = {'DBUsername': {'Type': 'String', 'Description': 'The WordPress database admin account username', 'Default': 'admin', 'MinLength': '1', 'AllowedPattern': '[a-zA-Z][a-zA-Z0-9]*', 'NoEcho': 'true', 'MaxLength': '16', 'ConstraintDescription': 'must begin with a letter and contain only alphanumeric characters.'}, 'KeyName': {'Type': 'String', 'Description': 'Name of an existing EC2 KeyPair to enable SSH access to the instances'}, 'LinuxDistribution': {'Default': 'F17', 'Type': 'String', 'Description': 'Distribution of choice', 'AllowedValues': ['F18', 'F17', 'U10', 'RHEL-6.1', 'RHEL-6.2', 'RHEL-6.3']}, 'DBPassword': {'Type': 'String', 'Description': 'The WordPress database admin account password', 'Default': 'admin', 'MinLength': '1', 'AllowedPattern': '[a-zA-Z0-9]*', 'NoEcho': 'true', 'MaxLength': '41', 'ConstraintDescription': 'must contain only alphanumeric characters.'}, 'DBName': {'AllowedPattern': '[a-zA-Z][a-zA-Z0-9]*', 'Type': 'String', 'Description': 'The WordPress database name', 'MaxLength': '64', 'Default': 'wordpress', 'MinLength': '1', 'ConstraintDescription': 'must begin with a letter and contain only alphanumeric characters.'}, 'InstanceType': {'Default': 'm1.large', 'Type': 'String', 'ConstraintDescription': 'must be a valid EC2 instance type.', 'Description': 'WebServer EC2 instance type', 'AllowedValues': ['t1.micro', 'm1.small', 'm1.large', 'm1.xlarge', 'm2.xlarge', 'm2.2xlarge', 'm2.4xlarge', 'c1.medium', 'c1.xlarge', 'cc1.4xlarge']}, 'DBRootPassword': {'Type': 'String', 'Description': 'Root password for MySQL', 'Default': 'admin', 'MinLength': '1', 'AllowedPattern': '[a-zA-Z0-9]*', 'NoEcho': 'true', 'MaxLength': '41', 'ConstraintDescription': 'must contain only alphanumeric characters.'}}
        expected = {'DBUsername': {'type': 'string', 'description': 'The WordPress database admin account username', 'required': False, 'update_allowed': True, 'immutable': False, 'constraints': [{'length': {'min': 1, 'max': 16}, 'description': 'must begin with a letter and contain only alphanumeric characters.'}, {'allowed_pattern': '[a-zA-Z][a-zA-Z0-9]*', 'description': 'must begin with a letter and contain only alphanumeric characters.'}]}, 'LinuxDistribution': {'type': 'string', 'description': 'Distribution of choice', 'required': False, 'update_allowed': True, 'immutable': False, 'constraints': [{'allowed_values': ['F18', 'F17', 'U10', 'RHEL-6.1', 'RHEL-6.2', 'RHEL-6.3']}]}, 'InstanceType': {'type': 'string', 'description': 'WebServer EC2 instance type', 'required': False, 'update_allowed': True, 'immutable': False, 'constraints': [{'allowed_values': ['t1.micro', 'm1.small', 'm1.large', 'm1.xlarge', 'm2.xlarge', 'm2.2xlarge', 'm2.4xlarge', 'c1.medium', 'c1.xlarge', 'cc1.4xlarge'], 'description': 'must be a valid EC2 instance type.'}]}, 'DBRootPassword': {'type': 'string', 'description': 'Root password for MySQL', 'required': False, 'update_allowed': True, 'immutable': False, 'constraints': [{'length': {'min': 1, 'max': 41}, 'description': 'must contain only alphanumeric characters.'}, {'allowed_pattern': '[a-zA-Z0-9]*', 'description': 'must contain only alphanumeric characters.'}]}, 'KeyName': {'type': 'string', 'description': 'Name of an existing EC2 KeyPair to enable SSH access to the instances', 'required': True, 'update_allowed': True, 'immutable': False}, 'DBPassword': {'type': 'string', 'description': 'The WordPress database admin account password', 'required': False, 'update_allowed': True, 'immutable': False, 'constraints': [{'length': {'min': 1, 'max': 41}, 'description': 'must contain only alphanumeric characters.'}, {'allowed_pattern': '[a-zA-Z0-9]*', 'description': 'must contain only alphanumeric characters.'}]}, 'DBName': {'type': 'string', 'description': 'The WordPress database name', 'required': False, 'update_allowed': True, 'immutable': False, 'constraints': [{'length': {'min': 1, 'max': 64}, 'description': 'must begin with a letter and contain only alphanumeric characters.'}, {'allowed_pattern': '[a-zA-Z][a-zA-Z0-9]*', 'description': 'must begin with a letter and contain only alphanumeric characters.'}]}}
        params = dict(((n, parameters.Schema.from_dict(n, s)) for n, s in params_snippet.items()))
        props_schemata = properties.Properties.schema_from_params(params)
        self.assertEqual(expected, dict(((n, dict(s)) for n, s in props_schemata.items())))

    def test_schema_from_hot_params(self):
        params_snippet = {'KeyName': {'type': 'string', 'description': 'Name of an existing EC2 KeyPair to enable SSH access to the instances'}, 'InstanceType': {'default': 'm1.large', 'type': 'string', 'description': 'WebServer EC2 instance type', 'constraints': [{'allowed_values': ['t1.micro', 'm1.small', 'm1.large', 'm1.xlarge', 'm2.xlarge', 'm2.2xlarge', 'm2.4xlarge', 'c1.medium', 'c1.xlarge', 'cc1.4xlarge'], 'description': 'Must be a valid EC2 instance type.'}]}, 'LinuxDistribution': {'default': 'F17', 'type': 'string', 'description': 'Distribution of choice', 'constraints': [{'allowed_values': ['F18', 'F17', 'U10', 'RHEL-6.1', 'RHEL-6.2', 'RHEL-6.3'], 'description': 'Must be a valid Linux distribution'}]}, 'DBName': {'type': 'string', 'description': 'The WordPress database name', 'default': 'wordpress', 'constraints': [{'length': {'min': 1, 'max': 64}, 'description': 'Length must be between 1 and 64'}, {'allowed_pattern': '[a-zA-Z][a-zA-Z0-9]*', 'description': 'Must begin with a letter and contain only alphanumeric characters.'}]}, 'DBUsername': {'type': 'string', 'description': 'The WordPress database admin account username', 'default': 'admin', 'hidden': 'true', 'constraints': [{'length': {'min': 1, 'max': 16}, 'description': 'Length must be between 1 and 16'}, {'allowed_pattern': '[a-zA-Z][a-zA-Z0-9]*', 'description': 'Must begin with a letter and only contain alphanumeric characters'}]}, 'DBPassword': {'type': 'string', 'description': 'The WordPress database admin account password', 'default': 'admin', 'hidden': 'true', 'constraints': [{'length': {'min': 1, 'max': 41}, 'description': 'Length must be between 1 and 41'}, {'allowed_pattern': '[a-zA-Z0-9]*', 'description': 'Must contain only alphanumeric characters'}]}, 'DBRootPassword': {'type': 'string', 'description': 'Root password for MySQL', 'default': 'admin', 'hidden': 'true', 'constraints': [{'length': {'min': 1, 'max': 41}, 'description': 'Length must be between 1 and 41'}, {'allowed_pattern': '[a-zA-Z0-9]*', 'description': 'Must contain only alphanumeric characters'}]}}
        expected = {'KeyName': {'type': 'string', 'description': 'Name of an existing EC2 KeyPair to enable SSH access to the instances', 'required': True, 'update_allowed': True, 'immutable': False}, 'InstanceType': {'type': 'string', 'description': 'WebServer EC2 instance type', 'required': False, 'update_allowed': True, 'immutable': False, 'constraints': [{'allowed_values': ['t1.micro', 'm1.small', 'm1.large', 'm1.xlarge', 'm2.xlarge', 'm2.2xlarge', 'm2.4xlarge', 'c1.medium', 'c1.xlarge', 'cc1.4xlarge'], 'description': 'Must be a valid EC2 instance type.'}]}, 'LinuxDistribution': {'type': 'string', 'description': 'Distribution of choice', 'required': False, 'update_allowed': True, 'immutable': False, 'constraints': [{'allowed_values': ['F18', 'F17', 'U10', 'RHEL-6.1', 'RHEL-6.2', 'RHEL-6.3'], 'description': 'Must be a valid Linux distribution'}]}, 'DBName': {'type': 'string', 'description': 'The WordPress database name', 'required': False, 'update_allowed': True, 'immutable': False, 'constraints': [{'length': {'min': 1, 'max': 64}, 'description': 'Length must be between 1 and 64'}, {'allowed_pattern': '[a-zA-Z][a-zA-Z0-9]*', 'description': 'Must begin with a letter and contain only alphanumeric characters.'}]}, 'DBUsername': {'type': 'string', 'description': 'The WordPress database admin account username', 'required': False, 'update_allowed': True, 'immutable': False, 'constraints': [{'length': {'min': 1, 'max': 16}, 'description': 'Length must be between 1 and 16'}, {'allowed_pattern': '[a-zA-Z][a-zA-Z0-9]*', 'description': 'Must begin with a letter and only contain alphanumeric characters'}]}, 'DBPassword': {'type': 'string', 'description': 'The WordPress database admin account password', 'required': False, 'update_allowed': True, 'immutable': False, 'constraints': [{'length': {'min': 1, 'max': 41}, 'description': 'Length must be between 1 and 41'}, {'allowed_pattern': '[a-zA-Z0-9]*', 'description': 'Must contain only alphanumeric characters'}]}, 'DBRootPassword': {'type': 'string', 'description': 'Root password for MySQL', 'required': False, 'update_allowed': True, 'immutable': False, 'constraints': [{'length': {'min': 1, 'max': 41}, 'description': 'Length must be between 1 and 41'}, {'allowed_pattern': '[a-zA-Z0-9]*', 'description': 'Must contain only alphanumeric characters'}]}}
        params = dict(((n, hot_param.HOTParamSchema.from_dict(n, s)) for n, s in params_snippet.items()))
        props_schemata = properties.Properties.schema_from_params(params)
        self.assertEqual(expected, dict(((n, dict(s)) for n, s in props_schemata.items())))

    def test_compare_same(self):
        schema = {'foo': {'Type': 'Integer'}}
        props_a = properties.Properties(schema, {'foo': 1})
        props_b = properties.Properties(schema, {'foo': 1})
        self.assertFalse(props_a != props_b)

    def test_compare_different(self):
        schema = {'foo': {'Type': 'Integer'}}
        props_a = properties.Properties(schema, {'foo': 0})
        props_b = properties.Properties(schema, {'foo': 1})
        self.assertTrue(props_a != props_b)

    def test_description_substitution(self):
        schema = {'description': properties.Schema('String', update_allowed=True), 'not_description': properties.Schema('String', update_allowed=True)}
        blank_rsrc = rsrc_defn.ResourceDefinition('foo', 'FooResource', {}, description='Foo resource')
        bar_rsrc = rsrc_defn.ResourceDefinition('foo', 'FooResource', {'description': 'bar'}, description='Foo resource')
        blank_props = blank_rsrc.properties(schema)
        self.assertEqual('Foo resource', blank_props['description'])
        self.assertEqual(None, blank_props['not_description'])
        replace_schema = {'description': properties.Schema('String')}
        empty_props = blank_rsrc.properties(replace_schema)
        self.assertEqual(None, empty_props['description'])
        bar_props = bar_rsrc.properties(schema)
        self.assertEqual('bar', bar_props['description'])

    def test_null_property_value(self):

        class NullFunction(function.Function):

            def result(self):
                return Ellipsis
        schema = {'Foo': properties.Schema('String', required=False), 'Bar': properties.Schema('String', required=False), 'Baz': properties.Schema('String', required=False)}
        user_props = {'Foo': NullFunction(None, 'null', []), 'Baz': None}
        props = properties.Properties(schema, user_props, function.resolve)
        self.assertEqual(None, props['Foo'])
        self.assertEqual(None, props.get_user_value('Foo'))
        self.assertEqual(None, props['Bar'])
        self.assertEqual(None, props.get_user_value('Bar'))
        self.assertEqual('', props['Baz'])
        self.assertEqual('', props.get_user_value('Baz'))