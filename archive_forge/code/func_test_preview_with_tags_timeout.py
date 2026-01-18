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
@mock.patch.object(rpc_client.EngineClient, 'call')
@mock.patch.object(stacks.stacks_view, 'format_stack')
def test_preview_with_tags_timeout(self, mock_format, mock_call, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'preview', True)
    identity = identifier.HeatIdentifier(self.tenant, 'wordpress', '1')
    template = {u'Foo': u'bar'}
    parameters = {u'InstanceType': u'm1.xlarge'}
    body = {'template': template, 'stack_name': identity.stack_name, 'parameters': parameters, 'tags': 'tag1,tag2', 'timeout_mins': 30}
    req = self._post('/stacks/preview', json.dumps(body))
    mock_call.return_value = {}
    mock_format.return_value = 'formatted_stack_preview'
    response = self.controller.preview(req, tenant_id=identity.tenant, body=body)
    rpc_client.EngineClient.call.assert_called_once_with(req.context, ('preview_stack', {'stack_name': identity.stack_name, 'template': template, 'params': {'parameters': parameters, 'encrypted_param_names': [], 'parameter_defaults': {}, 'event_sinks': [], 'resource_registry': {}}, 'files': {}, 'environment_files': None, 'files_container': None, 'args': {'timeout_mins': 30, 'tags': ['tag1', 'tag2']}}), version='1.36')
    self.assertEqual({'stack': 'formatted_stack_preview'}, response)