import json
from unittest import mock
from oslo_config import cfg
import webob.exc
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.stacks as stacks
from heat.api.openstack.v1.views import stacks_view
from heat.common import context
from heat.common import exception as heat_exc
from heat.common import identifier
from heat.common import policy
from heat.common import template_format
from heat.common import urlfetch
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
def test_resource_schema(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'resource_schema', True)
    req = self._get('/resource_types/ResourceWithProps')
    type_name = 'ResourceWithProps'
    engine_response = {'resource_type': type_name, 'properties': {'Foo': {'type': 'string', 'required': False}}, 'attributes': {'foo': {'description': 'A generic attribute'}, 'Foo': {'description': 'Another generic attribute'}}, 'support_status': {'status': 'SUPPORTED', 'version': None, 'message': None}}
    mock_call = self.patchobject(rpc_client.EngineClient, 'call', return_value=engine_response)
    response = self.controller.resource_schema(req, tenant_id=self.tenant, type_name=type_name)
    self.assertEqual(engine_response, response)
    mock_call.assert_called_once_with(req.context, ('resource_schema', {'type_name': type_name, 'with_description': False}), version='1.30')