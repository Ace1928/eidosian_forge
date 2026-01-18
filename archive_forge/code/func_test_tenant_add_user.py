import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
from keystoneclient.v2_0 import tenants
from keystoneclient.v2_0 import users
def test_tenant_add_user(self):
    self.stub_url('PUT', ['tenants', self.EXTRAS_ID, 'users', 'foo', 'roles', 'OS-KSADM', 'barrr'], status_code=204)
    req_body = {'tenant': {'id': self.EXTRAS_ID, 'name': 'tenantX', 'description': 'I changed you!', 'enabled': False}}
    tenant = self.client.tenants.resource_class(self.client.tenants, req_body['tenant'])
    tenant.add_user('foo', 'barrr')
    self.assertIsInstance(tenant, tenants.Tenant)