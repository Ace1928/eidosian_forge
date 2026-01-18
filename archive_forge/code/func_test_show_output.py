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
def test_show_output(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'show_output', True)
    identity = identifier.HeatIdentifier(self.tenant, 'wordpress', '6')
    req = self._get('/stacks/%(stack_name)s/%(stack_id)s/key' % identity)
    output = {'output_key': 'key', 'output_value': 'val', 'description': 'description'}
    mock_call = self.patchobject(rpc_client.EngineClient, 'call', return_value=output)
    response = self.controller.show_output(req, tenant_id=identity.tenant, stack_name=identity.stack_name, stack_id=identity.stack_id, output_key='key')
    self.assertEqual({'output': output}, response)
    mock_call.assert_called_once_with(req.context, ('show_output', {'output_key': 'key', 'stack_identity': dict(identity)}), version='1.19')