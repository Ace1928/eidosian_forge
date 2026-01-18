import uuid
from keystoneauth1 import exceptions
import testresources
from keystoneclient import access
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit.v3 import utils
def test_validate_token_allow_expired(self):
    token_id = uuid.uuid4().hex
    token_ref = self.examples.TOKEN_RESPONSES[self.examples.v3_UUID_TOKEN_UNSCOPED]
    self.stub_url('GET', ['auth', 'tokens'], headers={'X-Subject-Token': token_id}, json=token_ref)
    self.client.tokens.validate(token_id)
    self.assertQueryStringIs()
    self.client.tokens.validate(token_id, allow_expired=True)
    self.assertQueryStringIs('allow_expired=1')