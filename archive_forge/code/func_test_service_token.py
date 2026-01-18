import itertools
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystonemiddleware.auth_token import _request
from keystonemiddleware.tests.unit import utils
def test_service_token(self):
    token = uuid.uuid4().hex
    self.assertIsNone(self.request.service_token)
    self.request.headers['X-Service-Token'] = token
    self.assertEqual(token, self.request.service_token)