from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_secret_acl_remove(self, entity_ref=None):
    entity_ref = entity_ref or self.secret_ref
    self.responses.delete(self.secret_acl_ref)
    entity = self.manager.create(entity_ref=entity_ref, users=self.users2)
    api_resp = entity.remove()
    self.assertEqual(self.secret_acl_ref, self.responses.last_request.url)
    self.assertIsNone(api_resp)