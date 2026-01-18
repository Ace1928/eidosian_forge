import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_revoke_both_project_and_domain(self):
    uris = self._get_mock_role_query_urls(self.role_data, project_data=self.project_data, user_data=self.user_data, domain_data=self.domain_data, use_role_name=True, use_user_name=True, use_project_name=True, use_domain_name=True, use_domain_in_query=True)
    uris.extend([dict(method='HEAD', uri=self.get_mock_url(resource='projects', append=[self.project_data.project_id, 'users', self.user_data.user_id, 'roles', self.role_data.role_id]), complete_qs=True, status_code=204), dict(method='DELETE', uri=self.get_mock_url(resource='projects', append=[self.project_data.project_id, 'users', self.user_data.user_id, 'roles', self.role_data.role_id]), status_code=200)])
    self.register_uris(uris)
    self.assertTrue(self.cloud.revoke_role(self.role_data.role_name, user=self.user_data.name, project=self.project_data.project_name, domain=self.domain_data.domain_name))