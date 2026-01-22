import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def assertCreateV2(self, **kwargs):
    auth = self.new_plugin(**kwargs)
    auth_ref = auth.get_auth_ref(self.session)
    self.assertIsInstance(auth_ref, access.AccessInfoV2)
    self.assertEqual(self.TEST_URL + 'v2.0/tokens', self.requests_mock.last_request.url)
    self.assertIsInstance(auth._plugin, self.V2_PLUGIN_CLASS)
    return auth