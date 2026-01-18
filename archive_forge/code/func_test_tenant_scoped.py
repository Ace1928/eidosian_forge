import uuid
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_tenant_scoped(self):
    tenant_id = uuid.uuid4().hex
    tenant_name = uuid.uuid4().hex
    token = fixture.V2Token(tenant_id=tenant_id, tenant_name=tenant_name)
    self.assertEqual(tenant_id, token.tenant_id)
    self.assertEqual(tenant_id, token['access']['token']['tenant']['id'])
    self.assertEqual(tenant_name, token.tenant_name)
    tn = token['access']['token']['tenant']['name']
    self.assertEqual(tenant_name, tn)
    self.assertIsNone(token.trust_id)