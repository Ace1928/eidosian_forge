from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_add_operation_acl(self):
    entity = self.manager.create(entity_ref=self.secret_ref + '///', users=self.users1, project_access=True)
    self.assertIsInstance(entity, acls.SecretACL)
    entity.add_operation_acl(users=self.users2, project_access=False, operation_type='read')
    read_acl = entity.read
    self.assertEqual(self.secret_ref + '/acl', read_acl.acl_ref)
    self.assertFalse(read_acl.project_access)
    self.assertEqual(self.users2, read_acl.users)
    self.assertEqual(acls.DEFAULT_OPERATION_TYPE, read_acl.operation_type)
    entity.add_operation_acl(users=[], project_access=False, operation_type='dummy')
    dummy_acl = entity.get('dummy')
    self.assertFalse(dummy_acl.project_access)
    self.assertEqual([], dummy_acl.users)