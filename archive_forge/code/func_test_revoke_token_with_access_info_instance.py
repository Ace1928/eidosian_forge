import uuid
from keystoneauth1 import exceptions
import testresources
from keystoneclient import access
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit.v3 import utils
def test_revoke_token_with_access_info_instance(self):
    token_id = uuid.uuid4().hex
    token_ref = self.examples.TOKEN_RESPONSES[self.examples.v3_UUID_TOKEN_DEFAULT]
    token = access.AccessInfoV3(token_id, token_ref['token'])
    self.stub_url('DELETE', ['/auth/tokens'], status_code=204)
    self.client.tokens.revoke_token(token)
    self.assertRequestHeaderEqual('X-Subject-Token', token_id)