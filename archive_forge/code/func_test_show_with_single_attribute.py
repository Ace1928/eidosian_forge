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
def test_show_with_single_attribute(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'show', True)
    res_name = 'WikiDatabase'
    stack_identity = identifier.HeatIdentifier(self.tenant, 'foo', '1')
    res_identity = identifier.ResourceIdentifier(resource_name=res_name, **stack_identity)
    mock_describe = mock.Mock(return_value={'foo': 'bar'})
    self.controller.rpc_client.describe_stack_resource = mock_describe
    req = self._get(res_identity._tenant_path(), {'with_attr': 'baz'})
    resp = self.controller.show(req, tenant_id=self.tenant, stack_name=stack_identity.stack_name, stack_id=stack_identity.stack_id, resource_name=res_name)
    self.assertEqual({'resource': {'foo': 'bar'}}, resp)
    args, kwargs = mock_describe.call_args
    self.assertIn('baz', kwargs['with_attr'])