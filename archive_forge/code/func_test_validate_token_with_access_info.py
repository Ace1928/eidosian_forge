import uuid
from keystoneauth1 import exceptions
import testresources
from keystoneclient import access
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit.v3 import utils
def test_validate_token_with_access_info(self):
    token_id = uuid.uuid4().hex
    token_ref = self.examples.TOKEN_RESPONSES[self.examples.v3_UUID_TOKEN_DEFAULT]
    token = access.AccessInfoV3(token_id, token_ref['token'])
    self.stub_url('GET', ['auth', 'tokens'], headers={'X-Subject-Token': token_id}, json=token_ref)
    access_info = self.client.tokens.validate(token)
    self.assertRequestHeaderEqual('X-Subject-Token', token_id)
    self.assertIsInstance(access_info, access.AccessInfoV3)
    self.assertEqual(token_id, access_info.auth_token)