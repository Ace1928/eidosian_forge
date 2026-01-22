import io
from oslo_config import fixture as config
from oslotest import base as test_base
import webob
from oslo_middleware import sizelimit
class TestRequestBodySizeLimiter(test_base.BaseTestCase):

    def setUp(self):
        super(TestRequestBodySizeLimiter, self).setUp()
        self.useFixture(config.Config())

        @webob.dec.wsgify()
        def fake_app(req):
            return webob.Response(req.body)
        self.middleware = sizelimit.RequestBodySizeLimiter(fake_app)
        self.MAX_REQUEST_BODY_SIZE = self.middleware.oslo_conf.oslo_middleware.max_request_body_size
        self.request = webob.Request.blank('/', method='POST')

    def test_content_length_acceptable(self):
        self.request.headers['Content-Length'] = self.MAX_REQUEST_BODY_SIZE
        self.request.body = b'0' * self.MAX_REQUEST_BODY_SIZE
        response = self.request.get_response(self.middleware)
        self.assertEqual(200, response.status_int)

    def test_content_length_too_large(self):
        self.request.headers['Content-Length'] = self.MAX_REQUEST_BODY_SIZE + 1
        self.request.body = b'0' * (self.MAX_REQUEST_BODY_SIZE + 1)
        response = self.request.get_response(self.middleware)
        self.assertEqual(413, response.status_int)