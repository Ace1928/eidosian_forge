from openstack import exceptions
from openstack.network.v2 import quota as _quota
from openstack.tests.unit import base
def test_update_quotas(self):
    project = self._get_project_data()
    self.register_uris([dict(method='GET', uri=self.get_mock_url('identity', 'public', append=['v3', 'projects', project.project_id]), json={'project': project.json_response['project']}), self.get_nova_discovery_mock_dict(), dict(method='PUT', uri=self.get_mock_url('compute', 'public', append=['os-quota-sets', project.project_id]), json={'quota_set': fake_quota_set}, validate=dict(json={'quota_set': {'cores': 1, 'force': True}}))])
    self.cloud.set_compute_quotas(project.project_id, cores=1)
    self.assert_calls()