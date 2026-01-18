import http.client as http
from oslo_serialization import jsonutils
import webob
from glance.common import auth
from glance.common import exception
from glance.tests import utils
def test_required_creds(self):
    """
        Test that plugin created without required
        credential pieces raises an exception
        """
    bad_creds = [{}, {'username': 'user1', 'strategy': 'keystone', 'password': 'pass'}, {'password': 'pass', 'strategy': 'keystone', 'auth_url': 'http://localhost/v1'}, {'username': 'user1', 'strategy': 'keystone', 'auth_url': 'http://localhost/v1'}, {'username': 'user1', 'password': 'pass', 'auth_url': 'http://localhost/v1'}, {'username': 'user1', 'password': 'pass', 'strategy': 'keystone', 'auth_url': 'http://localhost/v2.0/'}, {'username': None, 'password': 'pass', 'auth_url': 'http://localhost/v2.0/'}, {'username': 'user1', 'password': 'pass', 'auth_url': 'http://localhost/v2.0/', 'tenant': None}]
    for creds in bad_creds:
        try:
            plugin = auth.KeystoneStrategy(creds)
            plugin.authenticate()
            self.fail('Failed to raise correct exception when supplying bad credentials: %r' % creds)
        except exception.MissingCredentialError:
            continue