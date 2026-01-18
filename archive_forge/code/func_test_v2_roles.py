import datetime
import uuid
from oslo_utils import timeutils
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_v2_roles(self):
    role_id = 'a'
    role_name = 'b'
    token = fixture.V2Token()
    token.set_scope()
    token.add_role(id=role_id, name=role_name)
    auth_ref = access.create(body=token)
    self.assertIsInstance(auth_ref, access.AccessInfoV2)
    self.assertEqual([role_id], auth_ref.role_ids)
    self.assertEqual([role_id], auth_ref._data['access']['metadata']['roles'])
    self.assertEqual([role_name], auth_ref.role_names)
    self.assertEqual([{'name': role_name}], auth_ref._data['access']['user']['roles'])