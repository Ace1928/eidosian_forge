import uuid
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1 import service_token
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_setting_service_token(self):
    self.session.get(self.TEST_URL)
    headers = self.requests_mock.last_request.headers
    self.assertEqual(self.user_token_id, headers['X-Auth-Token'])
    self.assertEqual(self.service_token_id, headers['X-Service-Token'])
    self.assertTrue(self.user_mock.called_once)
    self.assertTrue(self.service_mock.called_once)