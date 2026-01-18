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
def test_schema_invalid_parameters_stripped(self):
    schema = {'foo': {'Type': 'String', 'Required': True, 'Implemented': True}}
    prop_expected = {'foo': {'Ref': 'foo'}}
    param_expected = {'foo': {'Type': 'String'}}
    parameters, props = properties.Properties.schema_to_parameters_and_properties(schema)
    self.assertEqual(param_expected, parameters)
    self.assertEqual(prop_expected, props)