import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneclient import access
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
from keystoneclient.v2_0 import tokens
def test_validate_token_invalid_token(self):
    id_ = uuid.uuid4().hex
    self.stub_url('GET', ['tokens', id_], status_code=404)
    self.assertRaises(exceptions.NotFound, self.client.tokens.get_token_data, id_)
    self.assertRaises(exceptions.NotFound, self.client.tokens.validate, id_)