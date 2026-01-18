import urllib
import uuid
import warnings
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import oidc
from keystoneauth1 import session
from keystoneauth1.tests.unit import oidc_fixtures
from keystoneauth1.tests.unit import utils
def test_wrong_grant_type(self):
    self.requests_mock.get(self.DISCOVERY_URL, json={'grant_types_supported': ['foo', 'bar']})
    plugin = self.plugin.__class__(self.AUTH_URL, self.IDENTITY_PROVIDER, self.PROTOCOL, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, discovery_endpoint=self.DISCOVERY_URL)
    self.assertRaises(exceptions.OidcPluginNotSupported, plugin.get_unscoped_auth_ref, self.session)