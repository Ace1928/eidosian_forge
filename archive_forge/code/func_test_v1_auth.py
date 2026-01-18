import http.client as http
from oslo_serialization import jsonutils
import webob
from glance.common import auth
from glance.common import exception
from glance.tests import utils
def test_v1_auth(self):
    """Test v1 auth code paths"""

    def fake_do_request(cls, url, method, headers=None, body=None):
        if url.find('2.0') != -1:
            self.fail('Invalid v1.0 token path (%s)' % url)
        headers = headers or {}
        resp = webob.Response()
        if headers.get('X-Auth-User') != 'user1' or headers.get('X-Auth-Key') != 'pass':
            resp.status = http.UNAUTHORIZED
        else:
            resp.status = http.OK
            resp.headers.update({'x-image-management-url': 'example.com'})
        return (FakeResponse(resp), '')
    self.mock_object(auth.KeystoneStrategy, '_do_request', fake_do_request)
    unauthorized_creds = [{'username': 'wronguser', 'auth_url': 'http://localhost/badauthurl/', 'strategy': 'keystone', 'region': 'RegionOne', 'password': 'pass'}, {'username': 'user1', 'auth_url': 'http://localhost/badauthurl/', 'strategy': 'keystone', 'region': 'RegionOne', 'password': 'badpass'}]
    for creds in unauthorized_creds:
        try:
            plugin = auth.KeystoneStrategy(creds)
            plugin.authenticate()
            self.fail('Failed to raise NotAuthenticated when supplying bad credentials: %r' % creds)
        except exception.NotAuthenticated:
            continue
    no_strategy_creds = {'username': 'user1', 'auth_url': 'http://localhost/redirect/', 'password': 'pass', 'region': 'RegionOne'}
    try:
        plugin = auth.KeystoneStrategy(no_strategy_creds)
        plugin.authenticate()
        self.fail('Failed to raise MissingCredentialError when supplying no strategy: %r' % no_strategy_creds)
    except exception.MissingCredentialError:
        pass
    good_creds = [{'username': 'user1', 'auth_url': 'http://localhost/redirect/', 'password': 'pass', 'strategy': 'keystone', 'region': 'RegionOne'}]
    for creds in good_creds:
        plugin = auth.KeystoneStrategy(creds)
        self.assertIsNone(plugin.authenticate())
        self.assertEqual('example.com', plugin.management_url)
    for creds in good_creds:
        plugin = auth.KeystoneStrategy(creds, configure_via_auth=False)
        self.assertIsNone(plugin.authenticate())
        self.assertIsNone(plugin.management_url)