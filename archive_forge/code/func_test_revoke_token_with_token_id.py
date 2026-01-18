import uuid
from keystoneauth1 import exceptions
import testresources
from keystoneclient import access
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit.v3 import utils
def test_revoke_token_with_token_id(self):
    token_id = uuid.uuid4().hex
    self.stub_url('DELETE', ['/auth/tokens'], status_code=204)
    self.client.tokens.revoke_token(token_id)
    self.assertRequestHeaderEqual('X-Subject-Token', token_id)