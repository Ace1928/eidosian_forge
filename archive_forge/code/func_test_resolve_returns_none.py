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