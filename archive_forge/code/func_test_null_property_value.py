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