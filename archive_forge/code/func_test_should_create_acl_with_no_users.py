from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_create_acl_with_no_users(self):
    entity = self.manager.create(entity_ref=self.container_ref, users=[])
    read_acl = entity.read
    self.assertEqual([], read_acl.users)
    self.assertEqual(acls.DEFAULT_OPERATION_TYPE, read_acl.operation_type)
    self.assertIsNone(read_acl.project_access)
    read_acl_via_get = entity.get('read')
    self.assertEqual(read_acl, read_acl_via_get)