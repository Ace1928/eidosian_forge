import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneclient import access
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
from keystoneclient.v2_0 import tokens
def test_with_tenant_id(self):
    token_fixture = fixture.V2Token()
    token_fixture.set_scope()
    self.stub_auth(json=token_fixture)
    token_id = uuid.uuid4().hex
    tenant_id = uuid.uuid4().hex
    token_ref = self.client.tokens.authenticate(token=token_id, tenant_id=tenant_id)
    self.assertIsInstance(token_ref, tokens.Token)
    self.assertEqual(token_fixture.token_id, token_ref.id)
    self.assertEqual(token_fixture.expires_str, token_ref.expires)
    tenant_data = {'id': token_fixture.tenant_id, 'name': token_fixture.tenant_name}
    self.assertEqual(tenant_data, token_ref.tenant)
    req_body = {'auth': {'token': {'id': token_id}, 'tenantId': tenant_id}}
    self.assertRequestBodyIs(json=req_body)