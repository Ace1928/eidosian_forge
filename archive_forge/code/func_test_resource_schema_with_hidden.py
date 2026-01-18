from unittest import mock
from heat.common import exception
from heat.engine import environment
from heat.engine import resource as res
from heat.engine import service
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_resource_schema_with_hidden(self):
    type_name = 'ResourceWithHiddenPropertyAndAttribute'
    expected = {'resource_type': type_name, 'properties': {'supported': {'description': 'Supported property.', 'type': 'list', 'immutable': False, 'required': False, 'update_allowed': False}}, 'attributes': {'supported': {'description': 'Supported attribute.', 'type': 'string'}, 'show': {'description': 'Detailed information about resource.', 'type': 'map'}}, 'support_status': {'status': 'SUPPORTED', 'version': None, 'message': None, 'previous_status': None}}
    schema = self.eng.resource_schema(self.ctx, type_name=type_name)
    self.assertEqual(expected, schema)