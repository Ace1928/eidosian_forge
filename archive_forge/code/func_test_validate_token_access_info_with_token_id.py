import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneclient import access
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
from keystoneclient.v2_0 import tokens
def test_validate_token_access_info_with_token_id(self):
    token_id = uuid.uuid4().hex
    token_fixture = fixture.V2Token(token_id=token_id)
    self.stub_url('GET', ['tokens', token_id], json=token_fixture)
    access_info = self.client.tokens.validate_access_info(token_id)
    self.assertIsInstance(access_info, access.AccessInfoV2)
    self.assertEqual(token_id, access_info.auth_token)