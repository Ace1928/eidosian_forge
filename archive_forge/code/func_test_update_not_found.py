import json
from unittest import mock
import webob.exc
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.software_deployments as software_deployments
from heat.common import exception as heat_exc
from heat.common import policy
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
@mock.patch.object(policy.Enforcer, 'enforce')
def test_update_not_found(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'update')
    deployment_id = 'a45559cd-8736-4375-bc39-d6a7bb62ade2'
    req = self._put('/software_deployments/%s' % deployment_id, '{}')
    error = heat_exc.NotFound('Not found %s' % deployment_id)
    with mock.patch.object(self.controller.rpc_client, 'update_software_deployment', side_effect=tools.to_remote_error(error)):
        resp = tools.request_with_middleware(fault.FaultWrapper, self.controller.update, req, deployment_id=deployment_id, body={}, tenant_id=self.tenant)
        self.assertEqual(404, resp.json['code'])
        self.assertEqual('NotFound', resp.json['error']['type'])