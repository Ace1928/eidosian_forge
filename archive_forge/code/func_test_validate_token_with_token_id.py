import uuid
from keystoneauth1 import exceptions
import testresources
from keystoneclient import access
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit.v3 import utils
def test_validate_token_with_token_id(self):
    token_id = uuid.uuid4().hex
    token_ref = self.examples.TOKEN_RESPONSES[self.examples.v3_UUID_TOKEN_DEFAULT]
    self.stub_url('GET', ['auth', 'tokens'], headers={'X-Subject-Token': token_id}, json=token_ref)
    token_data = self.client.tokens.get_token_data(token_id)
    self.assertEqual(token_data, token_ref)
    access_info = self.client.tokens.validate(token_id)
    self.assertRequestHeaderEqual('X-Subject-Token', token_id)
    self.assertIsInstance(access_info, access.AccessInfoV3)
    self.assertEqual(token_id, access_info.auth_token)