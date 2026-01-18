import uuid
from keystoneauth1 import exceptions
import testresources
from keystoneclient import access
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit.v3 import utils
def test_validate_token_invalid(self):
    token_id = uuid.uuid4().hex
    self.stub_url('GET', ['auth', 'tokens'], status_code=404)
    self.assertRaises(exceptions.NotFound, self.client.tokens.get_token_data, token_id)
    self.assertRaises(exceptions.NotFound, self.client.tokens.validate, token_id)