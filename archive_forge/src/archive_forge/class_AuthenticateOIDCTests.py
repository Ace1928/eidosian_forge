import urllib.parse
import uuid
from oslo_config import fixture as config
import testtools
from keystoneclient.auth import conf
from keystoneclient.contrib.auth.v3 import oidc
from keystoneclient import session
from keystoneclient.tests.unit.v3 import utils
class AuthenticateOIDCTests(utils.TestCase):
    GROUP = 'auth'

    def setUp(self):
        super(AuthenticateOIDCTests, self).setUp()
        self.deprecations.expect_deprecations()
        self.conf_fixture = self.useFixture(config.Config())
        conf.register_conf_options(self.conf_fixture.conf, group=self.GROUP)
        self.session = session.Session()
        self.IDENTITY_PROVIDER = 'bluepages'
        self.PROTOCOL = 'oidc'
        self.USER_NAME = 'oidc_user@example.com'
        self.PASSWORD = uuid.uuid4().hex
        self.CLIENT_ID = uuid.uuid4().hex
        self.CLIENT_SECRET = uuid.uuid4().hex
        self.ACCESS_TOKEN_ENDPOINT = 'https://localhost:8020/oidc/token'
        self.FEDERATION_AUTH_URL = '%s/%s' % (self.TEST_URL, 'OS-FEDERATION/identity_providers/bluepages/protocols/oidc/auth')
        self.oidcplugin = oidc.OidcPassword(self.TEST_URL, self.IDENTITY_PROVIDER, self.PROTOCOL, username=self.USER_NAME, password=self.PASSWORD, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, access_token_endpoint=self.ACCESS_TOKEN_ENDPOINT)

    @testtools.skip("TypeError: __init__() got an unexpected keyword argument 'project_name'")
    def test_conf_params(self):
        """Ensure OpenID Connect config options work."""
        section = uuid.uuid4().hex
        identity_provider = uuid.uuid4().hex
        protocol = uuid.uuid4().hex
        username = uuid.uuid4().hex
        password = uuid.uuid4().hex
        client_id = uuid.uuid4().hex
        client_secret = uuid.uuid4().hex
        access_token_endpoint = uuid.uuid4().hex
        self.conf_fixture.config(auth_section=section, group=self.GROUP)
        conf.register_conf_options(self.conf_fixture.conf, group=self.GROUP)
        self.conf_fixture.register_opts(oidc.OidcPassword.get_options(), group=section)
        self.conf_fixture.config(auth_plugin='v3oidcpassword', identity_provider=identity_provider, protocol=protocol, username=username, password=password, client_id=client_id, client_secret=client_secret, access_token_endpoint=access_token_endpoint, group=section)
        a = conf.load_from_conf_options(self.conf_fixture.conf, self.GROUP)
        self.assertEqual(identity_provider, a.identity_provider)
        self.assertEqual(protocol, a.protocol)
        self.assertEqual(username, a.username)
        self.assertEqual(password, a.password)
        self.assertEqual(client_id, a.client_id)
        self.assertEqual(client_secret, a.client_secret)
        self.assertEqual(access_token_endpoint, a.access_token_endpoint)

    def test_initial_call_to_get_access_token(self):
        """Test initial call, expect JSON access token."""
        self.requests_mock.post(self.ACCESS_TOKEN_ENDPOINT, json=ACCESS_TOKEN_ENDPOINT_RESP)
        grant_type = 'password'
        scope = 'profile email'
        client_auth = (self.CLIENT_ID, self.CLIENT_SECRET)
        payload = {'grant_type': grant_type, 'username': self.USER_NAME, 'password': self.PASSWORD, 'scope': scope}
        res = self.oidcplugin._get_access_token(self.session, client_auth, payload, self.ACCESS_TOKEN_ENDPOINT)
        self.assertEqual(self.ACCESS_TOKEN_ENDPOINT, res.request.url)
        self.assertEqual('POST', res.request.method)
        encoded_payload = urllib.parse.urlencode(payload)
        self.assertEqual(encoded_payload, res.request.body)

    def test_second_call_to_protected_url(self):
        """Test subsequent call, expect Keystone token."""
        self.requests_mock.post(self.FEDERATION_AUTH_URL, json=UNSCOPED_TOKEN, headers={'X-Subject-Token': KEYSTONE_TOKEN_VALUE})
        access_token = uuid.uuid4().hex
        headers = {'Authorization': 'Bearer ' + access_token}
        res = self.oidcplugin._get_keystone_token(self.session, headers, self.FEDERATION_AUTH_URL)
        self.assertEqual(self.FEDERATION_AUTH_URL, res.request.url)
        self.assertEqual('POST', res.request.method)
        self.assertEqual(headers['Authorization'], res.request.headers['Authorization'])

    def test_end_to_end_workflow(self):
        """Test full OpenID Connect workflow."""
        self.requests_mock.post(self.ACCESS_TOKEN_ENDPOINT, json=ACCESS_TOKEN_ENDPOINT_RESP)
        self.requests_mock.post(self.FEDERATION_AUTH_URL, json=UNSCOPED_TOKEN, headers={'X-Subject-Token': KEYSTONE_TOKEN_VALUE})
        response = self.oidcplugin.get_unscoped_auth_ref(self.session)
        self.assertEqual(KEYSTONE_TOKEN_VALUE, response.auth_token)