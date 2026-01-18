import webob
from heat.common import noauth
from heat.tests import common
def test_request_with_no_tenant_in_url_or_auth_headers(self):
    req = webob.Request.blank('/')
    self.middleware(req.environ, self._start_fake_response)
    self.assertEqual(200, self.response_status)