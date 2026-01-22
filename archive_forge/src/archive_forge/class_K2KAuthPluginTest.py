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
class K2KAuthPluginTest(utils.TestCase):
    TEST_ROOT_URL = 'http://127.0.0.1:5000/'
    TEST_URL = '%s%s' % (TEST_ROOT_URL, 'v3')
    TEST_PASS = 'password'
    REQUEST_ECP_URL = TEST_URL + '/auth/OS-FEDERATION/saml2/ecp'
    SP_ROOT_URL = 'https://sp1.com/v3'
    SP_ID = 'sp1'
    SP_URL = 'https://sp1.com/Shibboleth.sso/SAML2/ECP'
    SP_AUTH_URL = SP_ROOT_URL + '/OS-FEDERATION/identity_providers/testidp/protocols/saml2/auth'
    SERVICE_PROVIDER_DICT = {'id': SP_ID, 'auth_url': SP_AUTH_URL, 'sp_url': SP_URL}

    def setUp(self):
        super(K2KAuthPluginTest, self).setUp()
        self.token_v3 = fixture.V3Token()
        self.token_v3.add_service_provider(self.SP_ID, self.SP_AUTH_URL, self.SP_URL)
        self.session = session.Session()
        self.k2kplugin = self.get_plugin()

    def _get_base_plugin(self):
        self.stub_url('POST', ['auth', 'tokens'], headers={'X-Subject-Token': uuid.uuid4().hex}, json=self.token_v3)
        return v3.Password(self.TEST_URL, username=self.TEST_USER, password=self.TEST_PASS)

    def _mock_k2k_flow_urls(self, redirect_code=302):
        self.requests_mock.get(self.TEST_URL, json={'version': fixture.V3Discovery(self.TEST_URL)}, headers={'Content-Type': 'application/json'})
        self.requests_mock.register_uri('POST', self.REQUEST_ECP_URL, content=bytes(k2k_fixtures.ECP_ENVELOPE, 'latin-1'), headers={'Content-Type': 'application/vnd.paos+xml'}, status_code=200)
        self.requests_mock.register_uri('POST', self.SP_URL, content=bytes(k2k_fixtures.TOKEN_BASED_ECP, 'latin-1'), headers={'Content-Type': 'application/vnd.paos+xml'}, status_code=redirect_code)
        self.requests_mock.register_uri('GET', self.SP_AUTH_URL, json=k2k_fixtures.UNSCOPED_TOKEN, headers={'X-Subject-Token': k2k_fixtures.UNSCOPED_TOKEN_HEADER})

    def get_plugin(self, **kwargs):
        kwargs.setdefault('base_plugin', self._get_base_plugin())
        kwargs.setdefault('service_provider', self.SP_ID)
        return v3.Keystone2Keystone(**kwargs)

    def test_remote_url(self):
        remote_auth_url = self.k2kplugin._remote_auth_url(self.SP_AUTH_URL)
        self.assertEqual(self.SP_ROOT_URL, remote_auth_url)

    def test_fail_getting_ecp_assertion(self):
        self.requests_mock.get(self.TEST_URL, json={'version': fixture.V3Discovery(self.TEST_URL)}, headers={'Content-Type': 'application/json'})
        self.requests_mock.register_uri('POST', self.REQUEST_ECP_URL, status_code=401)
        self.assertRaises(exceptions.AuthorizationFailure, self.k2kplugin._get_ecp_assertion, self.session)

    def test_get_ecp_assertion_empty_response(self):
        self.requests_mock.get(self.TEST_URL, json={'version': fixture.V3Discovery(self.TEST_URL)}, headers={'Content-Type': 'application/json'})
        self.requests_mock.register_uri('POST', self.REQUEST_ECP_URL, headers={'Content-Type': 'application/vnd.paos+xml'}, content=b'', status_code=200)
        self.assertRaises(exceptions.InvalidResponse, self.k2kplugin._get_ecp_assertion, self.session)

    def test_get_ecp_assertion_wrong_headers(self):
        self.requests_mock.get(self.TEST_URL, json={'version': fixture.V3Discovery(self.TEST_URL)}, headers={'Content-Type': 'application/json'})
        self.requests_mock.register_uri('POST', self.REQUEST_ECP_URL, headers={'Content-Type': uuid.uuid4().hex}, content=b'', status_code=200)
        self.assertRaises(exceptions.InvalidResponse, self.k2kplugin._get_ecp_assertion, self.session)

    def test_send_ecp_authn_response(self):
        self._mock_k2k_flow_urls()
        response = self.k2kplugin._send_service_provider_ecp_authn_response(self.session, self.SP_URL, self.SP_AUTH_URL)
        self.assertEqual(k2k_fixtures.UNSCOPED_TOKEN_HEADER, response.headers['X-Subject-Token'])

    def test_send_ecp_authn_response_303_redirect(self):
        self._mock_k2k_flow_urls(redirect_code=303)
        response = self.k2kplugin._send_service_provider_ecp_authn_response(self.session, self.SP_URL, self.SP_AUTH_URL)
        self.assertEqual(k2k_fixtures.UNSCOPED_TOKEN_HEADER, response.headers['X-Subject-Token'])

    def test_end_to_end_workflow(self):
        self._mock_k2k_flow_urls()
        auth_ref = self.k2kplugin.get_auth_ref(self.session)
        self.assertEqual(k2k_fixtures.UNSCOPED_TOKEN_HEADER, auth_ref.auth_token)

    def test_end_to_end_workflow_303_redirect(self):
        self._mock_k2k_flow_urls(redirect_code=303)
        auth_ref = self.k2kplugin.get_auth_ref(self.session)
        self.assertEqual(k2k_fixtures.UNSCOPED_TOKEN_HEADER, auth_ref.auth_token)

    def test_end_to_end_with_generic_password(self):
        self.requests_mock.get(self.TEST_ROOT_URL, json=fixture.DiscoveryList(self.TEST_ROOT_URL), headers={'Content-Type': 'application/json'})
        self.requests_mock.register_uri('POST', self.REQUEST_ECP_URL, content=bytes(k2k_fixtures.ECP_ENVELOPE, 'latin-1'), headers={'Content-Type': 'application/vnd.paos+xml'}, status_code=200)
        self.requests_mock.register_uri('POST', self.SP_URL, content=bytes(k2k_fixtures.TOKEN_BASED_ECP, 'latin-1'), headers={'Content-Type': 'application/vnd.paos+xml'}, status_code=302)
        self.requests_mock.register_uri('GET', self.SP_AUTH_URL, json=k2k_fixtures.UNSCOPED_TOKEN, headers={'X-Subject-Token': k2k_fixtures.UNSCOPED_TOKEN_HEADER})
        self.stub_url('POST', ['auth', 'tokens'], headers={'X-Subject-Token': uuid.uuid4().hex}, json=self.token_v3)
        plugin = identity.Password(self.TEST_ROOT_URL, username=self.TEST_USER, password=self.TEST_PASS, user_domain_id='default')
        k2kplugin = self.get_plugin(base_plugin=plugin)
        self.assertEqual(k2k_fixtures.UNSCOPED_TOKEN_HEADER, k2kplugin.get_token(self.session))