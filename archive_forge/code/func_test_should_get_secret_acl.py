from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_get_secret_acl(self, entity_ref=None):
    entity_ref = entity_ref or self.secret_ref
    self.responses.get(self.secret_acl_ref, json=self.get_acl_response_data())
    api_resp = self.manager.get(entity_ref=entity_ref)
    self.assertEqual(self.secret_acl_ref, self.responses.last_request.url)
    self.assertFalse(api_resp.get('read').project_access)
    self.assertEqual('read', api_resp.get('read').operation_type)
    self.assertIn(api_resp.get('read').acl_ref_relative, self.secret_acl_ref)