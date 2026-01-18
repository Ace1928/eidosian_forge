from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_submit_acl_with_users_project_access_set(self, href=None):
    href = href or self.secret_ref
    data = {'acl_ref': self.secret_acl_ref}
    self.responses.put(self.secret_acl_ref, json=data)
    entity = self.manager.create(entity_ref=href + '///', users=self.users1, project_access=True)
    api_resp = entity.submit()
    self.assertEqual(self.secret_acl_ref, api_resp)
    self.assertEqual(self.secret_acl_ref, self.responses.last_request.url)