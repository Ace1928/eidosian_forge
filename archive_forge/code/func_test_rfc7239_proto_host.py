from wsgiref import util
from oslotest import base as test_base
import webob
from oslo_middleware import http_proxy_to_wsgi
def test_rfc7239_proto_host(self):
    self.request.headers['Forwarded'] = 'for=foobar;proto=https;host=example.com, for=foobaz;proto=http'
    response = self.request.get_response(self.middleware)
    self.assertEqual(b'https://example.com/', response.body)