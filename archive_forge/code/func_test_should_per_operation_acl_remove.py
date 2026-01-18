from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_per_operation_acl_remove(self):
    self.responses.get(self.secret_acl_ref, json=self.get_acl_response_data(users=self.users2, project_access=True))
    self.responses.delete(self.secret_acl_ref)
    entity = self.manager.create(entity_ref=self.secret_ref, users=self.users2)
    api_resp = entity.read.remove()
    self.assertEqual(self.secret_acl_ref, self.responses.last_request.url)
    self.assertIsNone(api_resp)
    self.assertEqual(0, len(entity.operation_acls))
    acl_data = self.get_acl_response_data(users=self.users2, project_access=True)
    data = self.get_acl_response_data(users=self.users1, operation_type='write', project_access=False)
    acl_data['write'] = data['write']
    self.responses.get(self.secret_acl_ref, json=acl_data)
    self.responses.put(self.secret_acl_ref, json={})
    entity = self.manager.create(entity_ref=self.secret_ref, users=self.users2)
    entity.read.remove()
    self.assertEqual(self.secret_acl_ref, self.responses.last_request.url)
    self.assertEqual(1, len(entity.operation_acls))
    self.assertEqual('write', entity.operation_acls[0].operation_type)
    self.assertEqual(self.users1, entity.operation_acls[0].users)