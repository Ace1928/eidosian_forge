import io
import uuid
from keystoneauth1 import fixture
from keystoneauth1 import plugin as ksa_plugin
from keystoneauth1 import session
from oslo_log import log as logging
from requests_mock.contrib import fixture as rm_fixture
from keystonemiddleware.auth_token import _auth
from keystonemiddleware.tests.unit import utils
def test_with_admin_token(self):
    token = uuid.uuid4().hex
    plugin = self.new_plugin(identity_uri='http://testhost:8888/admin', admin_token=token)
    self.assertEqual(token, plugin.get_token(self.session))