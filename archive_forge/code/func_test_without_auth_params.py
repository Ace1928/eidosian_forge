import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneclient import access
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
from keystoneclient.v2_0 import tokens
def test_without_auth_params(self):
    self.assertRaises(ValueError, self.client.tokens.authenticate)
    self.assertRaises(ValueError, self.client.tokens.authenticate, tenant_id=uuid.uuid4().hex)