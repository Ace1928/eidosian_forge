from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_create_acl_with_users(self, entity_ref=None):
    entity_ref = entity_ref or self.container_ref
    entity = self.manager.create(entity_ref=entity_ref + '///', users=self.users2, project_access=False)
    self.assertIsInstance(entity, acls.ContainerACL)
    self.assertEqual(entity_ref + '///', entity.entity_ref)
    read_acl = entity.read
    self.assertFalse(read_acl.project_access)
    self.assertEqual(self.users2, read_acl.users)
    self.assertEqual(acls.DEFAULT_OPERATION_TYPE, read_acl.operation_type)
    self.assertIn(entity_ref, read_acl.acl_ref, 'ACL ref has additional /acl')
    self.assertIn(read_acl.acl_ref_relative, self.container_acl_ref)