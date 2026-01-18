from unittest import mock
import webob.exc
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.resources as resources
from heat.common import exception as heat_exc
from heat.common import identifier
from heat.common import policy
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
def test_show_with_nested_stack(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'show', True)
    res_name = 'WikiDatabase'
    stack_identity = identifier.HeatIdentifier(self.tenant, 'wordpress', '6')
    res_identity = identifier.ResourceIdentifier(resource_name=res_name, **stack_identity)
    nested_stack_identity = identifier.HeatIdentifier(self.tenant, 'nested', 'some_id')
    req = self._get(stack_identity._tenant_path())
    engine_resp = {u'description': u'', u'resource_identity': dict(res_identity), u'stack_name': stack_identity.stack_name, u'resource_name': res_name, u'resource_status_reason': None, u'updated_time': u'2012-07-23T13:06:00Z', u'stack_identity': dict(stack_identity), u'resource_action': u'CREATE', u'resource_status': u'COMPLETE', u'physical_resource_id': u'a3455d8c-9f88-404d-a85b-5315293e67de', u'resource_type': u'AWS::EC2::Instance', u'attributes': {u'foo': 'bar'}, u'metadata': {u'ensureRunning': u'true'}, u'nested_stack_id': dict(nested_stack_identity)}
    mock_call = self.patchobject(rpc_client.EngineClient, 'call', return_value=engine_resp)
    result = self.controller.show(req, tenant_id=self.tenant, stack_name=stack_identity.stack_name, stack_id=stack_identity.stack_id, resource_name=res_name)
    expected = [{'href': self._url(res_identity), 'rel': 'self'}, {'href': self._url(stack_identity), 'rel': 'stack'}, {'href': self._url(nested_stack_identity), 'rel': 'nested'}]
    self.assertEqual(expected, result['resource']['links'])
    self.assertIsNone(result.get(rpc_api.RES_NESTED_STACK_ID))
    mock_call.assert_called_once_with(req.context, ('describe_stack_resource', {'stack_identity': stack_identity, 'resource_name': res_name, 'with_attr': None}), version='1.2')