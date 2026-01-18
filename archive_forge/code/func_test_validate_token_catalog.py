import uuid
from keystoneauth1 import exceptions
import testresources
from keystoneclient import access
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit.v3 import utils
def test_validate_token_catalog(self):
    token_id = uuid.uuid4().hex
    token_ref = self.examples.TOKEN_RESPONSES[self.examples.v3_UUID_TOKEN_DEFAULT]
    self.stub_url('GET', ['auth', 'tokens'], headers={'X-Subject-Token': token_id}, json=token_ref)
    token_data = self.client.tokens.get_token_data(token_id)
    self.assertQueryStringIs()
    self.assertIn('catalog', token_data['token'])
    access_info = self.client.tokens.validate(token_id)
    self.assertQueryStringIs()
    self.assertTrue(access_info.has_service_catalog())