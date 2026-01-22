import webob
from heat.common import noauth
from heat.tests import common
class KeystonePasswordAuthProtocolTest(common.HeatTestCase):

    def setUp(self):
        super(KeystonePasswordAuthProtocolTest, self).setUp()
        self.config = {'auth_uri': 'http://keystone.test.com:5000'}
        self.app = FakeApp()
        self.middleware = noauth.NoAuthProtocol(self.app, self.config)

    def _start_fake_response(self, status, headers):
        self.response_status = int(status.split(' ', 1)[0])
        self.response_headers = dict(headers)

    def test_request_with_bad_credentials(self):
        req = webob.Request.blank('/tenant_id1/')
        req.headers['X_AUTH_USER'] = 'admin'
        req.headers['X_AUTH_KEY'] = 'blah'
        req.headers['X_AUTH_URL'] = self.config['auth_uri']
        self.middleware(req.environ, self._start_fake_response)
        self.assertEqual(200, self.response_status)

    def test_request_with_no_tenant_in_url_or_auth_headers(self):
        req = webob.Request.blank('/')
        self.middleware(req.environ, self._start_fake_response)
        self.assertEqual(200, self.response_status)