import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_v3_plugin_from_failure(self):
    url = self.TEST_URL + 'v3'
    self.stub_url('GET', [], base_url=url, status_code=403)
    self.assertCreateV3(auth_url=url)