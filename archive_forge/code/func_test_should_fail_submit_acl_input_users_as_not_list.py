from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_fail_submit_acl_input_users_as_not_list(self):
    data = {'acl_ref': self.secret_acl_ref}
    self.responses.put(self.secret_acl_ref, json=data)
    entity = self.manager.create(entity_ref=self.secret_ref, users='u1')
    self.assertRaises(ValueError, entity.submit)