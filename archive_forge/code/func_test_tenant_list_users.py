import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
from keystoneclient.v2_0 import tenants
from keystoneclient.v2_0 import users
def test_tenant_list_users(self):
    tenant_id = uuid.uuid4().hex
    user_id1 = uuid.uuid4().hex
    user_id2 = uuid.uuid4().hex
    tenant_resp = {'tenant': {'name': uuid.uuid4().hex, 'enabled': True, 'id': tenant_id, 'description': 'test tenant'}}
    users_resp = {'users': {'values': [{'email': uuid.uuid4().hex, 'enabled': True, 'id': user_id1, 'name': uuid.uuid4().hex}, {'email': uuid.uuid4().hex, 'enabled': True, 'id': user_id2, 'name': uuid.uuid4().hex}]}}
    self.stub_url('GET', ['tenants', tenant_id], json=tenant_resp)
    self.stub_url('GET', ['tenants', tenant_id, 'users'], json=users_resp)
    tenant = self.client.tenants.get(tenant_id)
    user_objs = tenant.list_users()
    for u in user_objs:
        self.assertIsInstance(u, users.User)
    self.assertEqual(set([user_id1, user_id2]), set([u.id for u in user_objs]))