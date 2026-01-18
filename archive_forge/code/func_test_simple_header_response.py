from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
def test_simple_header_response(self):
    """CORS Specification Section 3

        A header is said to be a simple header if the header field name is an
        ASCII case-insensitive match for Accept, Accept-Language, or
        Content-Language or if it is an ASCII case-insensitive match for
        Content-Type and the header field value media type (excluding
        parameters) is an ASCII case-insensitive match for
        application/x-www-form-urlencoded, multipart/form-data, or text/plain.

        NOTE: We are not testing the media type cases.
        """
    simple_headers = ','.join(['accept', 'accept-language', 'content-language', 'content-type'])
    request = webob.Request.blank('/')
    request.method = 'OPTIONS'
    request.headers['Origin'] = 'http://valid.example.com'
    request.headers['Access-Control-Request-Method'] = 'GET'
    request.headers['Access-Control-Request-Headers'] = simple_headers
    response = request.get_response(self.application)
    self.assertCORSResponse(response, status='200 OK', allow_origin='http://valid.example.com', max_age=None, allow_methods='GET', allow_headers=simple_headers, allow_credentials=None, expose_headers=None)