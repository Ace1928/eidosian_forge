from unittest import mock
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.build_info as build_info
from heat.common import policy
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
def test_retrieves_build_revision_from_the_engine(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'build_info', True)
    req = self._get('/build_info')
    mock_engine = mock.Mock()
    mock_engine.get_revision.return_value = 'engine_revision'
    self.controller.rpc_client = mock_engine
    response = self.controller.build_info(req, tenant_id=self.tenant)
    self.assertIn('engine', response)
    self.assertIn('revision', response['engine'])
    self.assertEqual('engine_revision', response['engine']['revision'])