import webob
from heat.common import noauth
from heat.tests import common
def test_request_with_bad_credentials(self):
    req = webob.Request.blank('/tenant_id1/')
    req.headers['X_AUTH_USER'] = 'admin'
    req.headers['X_AUTH_KEY'] = 'blah'
    req.headers['X_AUTH_URL'] = self.config['auth_uri']
    self.middleware(req.environ, self._start_fake_response)
    self.assertEqual(200, self.response_status)