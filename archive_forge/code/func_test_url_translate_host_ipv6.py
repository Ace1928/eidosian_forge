from wsgiref import util
from oslotest import base as test_base
import webob
from oslo_middleware import http_proxy_to_wsgi
def test_url_translate_host_ipv6(self):
    self.request.headers['X-Forwarded-Proto'] = 'https'
    self.request.headers['X-Forwarded-Host'] = '[f00:b4d::1]:123'
    response = self.request.get_response(self.middleware)
    self.assertEqual(b'https://[f00:b4d::1]:123/', response.body)