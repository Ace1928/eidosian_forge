import itertools
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystonemiddleware.auth_token import _request
from keystonemiddleware.tests.unit import utils
def test_setting_service_token_valid(self):
    self.assertNotIn('X-Service-Identity-Status', self.request.headers)
    self.request.service_token_valid = True
    self.assertEqual('Confirmed', self.request.headers['X-Service-Identity-Status'])
    self.assertTrue(self.request.service_token_valid)
    self.request.service_token_valid = False
    self.assertEqual('Invalid', self.request.headers['X-Service-Identity-Status'])
    self.assertFalse(self.request.service_token_valid)