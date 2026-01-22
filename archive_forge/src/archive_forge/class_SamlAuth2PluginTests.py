import base64
import uuid
import requests
from keystoneauth1 import exceptions
from keystoneauth1.extras import _saml2 as saml2
from keystoneauth1 import fixture as ksa_fixtures
from keystoneauth1 import session
from keystoneauth1.tests.unit.extras.saml2 import fixtures as saml2_fixtures
from keystoneauth1.tests.unit.extras.saml2 import utils
from keystoneauth1.tests.unit import matchers
class SamlAuth2PluginTests(utils.TestCase):
    """These test ONLY the standalone requests auth plugin.

    Tests for the auth plugin are later so that hopefully these can be
    extracted into it's own module.
    """
    HEADER_MEDIA_TYPE_SEPARATOR = ','
    TEST_USER = 'user'
    TEST_PASS = 'pass'
    TEST_SP_URL = 'http://sp.test'
    TEST_IDP_URL = 'http://idp.test'
    TEST_CONSUMER_URL = 'https://openstack4.local/Shibboleth.sso/SAML2/ECP'

    def get_plugin(self, **kwargs):
        kwargs.setdefault('identity_provider_url', self.TEST_IDP_URL)
        kwargs.setdefault('requests_auth', (self.TEST_USER, self.TEST_PASS))
        return saml2.v3.saml2._SamlAuth(**kwargs)

    @property
    def calls(self):
        return [r.url.strip('/') for r in self.requests_mock.request_history]

    def basic_header(self, username=TEST_USER, password=TEST_PASS):
        user_pass = ('%s:%s' % (username, password)).encode('utf-8')
        return 'Basic %s' % base64.b64encode(user_pass).decode('utf-8')

    def test_request_accept_headers(self):
        random_header = uuid.uuid4().hex
        headers = {'Accept': random_header}
        req = requests.Request('GET', 'http://another.test', headers=headers)
        plugin = self.get_plugin()
        plugin_headers = plugin(req).headers
        self.assertIn('Accept', plugin_headers)
        accept_header = plugin_headers['Accept']
        self.assertIn(self.HEADER_MEDIA_TYPE_SEPARATOR, accept_header)
        self.assertIn(random_header, accept_header.split(self.HEADER_MEDIA_TYPE_SEPARATOR))
        self.assertIn(PAOS_HEADER, accept_header.split(self.HEADER_MEDIA_TYPE_SEPARATOR))

    def test_passed_when_not_200(self):
        text = uuid.uuid4().hex
        test_url = 'http://another.test'
        self.requests_mock.get(test_url, status_code=201, headers=CONTENT_TYPE_PAOS_HEADER, text=text)
        resp = requests.get(test_url, auth=self.get_plugin())
        self.assertEqual(201, resp.status_code)
        self.assertEqual(text, resp.text)

    def test_200_without_paos_header(self):
        text = uuid.uuid4().hex
        test_url = 'http://another.test'
        self.requests_mock.get(test_url, status_code=200, text=text)
        resp = requests.get(test_url, auth=self.get_plugin())
        self.assertEqual(200, resp.status_code)
        self.assertEqual(text, resp.text)

    def test_standard_workflow_302_redirect(self):
        text = uuid.uuid4().hex
        self.requests_mock.get(self.TEST_SP_URL, response_list=[dict(headers=CONTENT_TYPE_PAOS_HEADER, content=utils.make_oneline(saml2_fixtures.SP_SOAP_RESPONSE)), dict(text=text)])
        authm = self.requests_mock.post(self.TEST_IDP_URL, content=saml2_fixtures.SAML2_ASSERTION)
        self.requests_mock.post(self.TEST_CONSUMER_URL, status_code=302, headers={'Location': self.TEST_SP_URL})
        resp = requests.get(self.TEST_SP_URL, auth=self.get_plugin())
        self.assertEqual(200, resp.status_code)
        self.assertEqual(text, resp.text)
        self.assertEqual(self.calls, [self.TEST_SP_URL, self.TEST_IDP_URL, self.TEST_CONSUMER_URL, self.TEST_SP_URL])
        self.assertEqual(self.basic_header(), authm.last_request.headers['Authorization'])
        authn_request = self.requests_mock.request_history[1].text
        self.assertThat(saml2_fixtures.AUTHN_REQUEST, matchers.XMLEquals(authn_request))

    def test_standard_workflow_303_redirect(self):
        text = uuid.uuid4().hex
        self.requests_mock.get(self.TEST_SP_URL, response_list=[dict(headers=CONTENT_TYPE_PAOS_HEADER, content=utils.make_oneline(saml2_fixtures.SP_SOAP_RESPONSE)), dict(text=text)])
        authm = self.requests_mock.post(self.TEST_IDP_URL, content=saml2_fixtures.SAML2_ASSERTION)
        self.requests_mock.post(self.TEST_CONSUMER_URL, status_code=303, headers={'Location': self.TEST_SP_URL})
        resp = requests.get(self.TEST_SP_URL, auth=self.get_plugin())
        self.assertEqual(200, resp.status_code)
        self.assertEqual(text, resp.text)
        url_flow = [self.TEST_SP_URL, self.TEST_IDP_URL, self.TEST_CONSUMER_URL, self.TEST_SP_URL]
        self.assertEqual(url_flow, [r.url.rstrip('/') for r in resp.history])
        self.assertEqual(url_flow, self.calls)
        self.assertEqual(self.basic_header(), authm.last_request.headers['Authorization'])
        authn_request = self.requests_mock.request_history[1].text
        self.assertThat(saml2_fixtures.AUTHN_REQUEST, matchers.XMLEquals(authn_request))

    def test_initial_sp_call_invalid_response(self):
        """Send initial SP HTTP request and receive wrong server response."""
        self.requests_mock.get(self.TEST_SP_URL, headers=CONTENT_TYPE_PAOS_HEADER, text='NON XML RESPONSE')
        self.assertRaises(InvalidResponse, requests.get, self.TEST_SP_URL, auth=self.get_plugin())
        self.assertEqual(self.calls, [self.TEST_SP_URL])

    def test_consumer_mismatch_error_workflow(self):
        consumer1 = 'http://consumer1/Shibboleth.sso/SAML2/ECP'
        consumer2 = 'http://consumer2/Shibboleth.sso/SAML2/ECP'
        soap_response = saml2_fixtures.soap_response(consumer=consumer1)
        saml_assertion = saml2_fixtures.saml_assertion(destination=consumer2)
        self.requests_mock.get(self.TEST_SP_URL, headers=CONTENT_TYPE_PAOS_HEADER, content=soap_response)
        self.requests_mock.post(self.TEST_IDP_URL, content=saml_assertion)
        saml_error = self.requests_mock.post(consumer1)
        self.assertRaises(saml2.v3.saml2.ConsumerMismatch, requests.get, self.TEST_SP_URL, auth=self.get_plugin())
        self.assertTrue(saml_error.called)