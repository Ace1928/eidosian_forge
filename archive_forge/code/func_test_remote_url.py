import copy
import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1.identity import v3
from keystoneauth1 import session
from keystoneauth1.tests.unit import k2k_fixtures
from keystoneauth1.tests.unit import utils
def test_remote_url(self):
    remote_auth_url = self.k2kplugin._remote_auth_url(self.SP_AUTH_URL)
    self.assertEqual(self.SP_ROOT_URL, remote_auth_url)