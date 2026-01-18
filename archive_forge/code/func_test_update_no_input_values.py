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
def test_update_no_input_values(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'update')
    config_id = 'd00ba4aa-db33-42e1-92f4-2a6469260107'
    server_id = 'fb322564-7927-473d-8aad-68ae7fbf2abf'
    body = {'action': 'INIT', 'status': 'COMPLETE', 'status_reason': None, 'config_id': config_id}
    return_value = body.copy()
    deployment_id = 'a45559cd-8736-4375-bc39-d6a7bb62ade2'
    return_value['id'] = deployment_id
    req = self._put('/software_deployments/%s' % deployment_id, json.dumps(body))
    return_value['server_id'] = server_id
    expected = {'software_deployment': return_value}
    with mock.patch.object(self.controller.rpc_client, 'update_software_deployment', return_value=return_value):
        resp = self.controller.update(req, deployment_id=deployment_id, body=body, tenant_id=self.tenant)
        self.assertEqual(expected, resp)