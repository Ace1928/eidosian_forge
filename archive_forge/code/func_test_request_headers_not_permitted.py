from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
def test_request_headers_not_permitted(self):
    """CORS Specification Section 6.2.4, 6.2.6

        If there are no Access-Control-Request-Headers headers let header
        field-names be the empty list.

        If any of the header field-names is not a ASCII case-insensitive
        match for any of the values in list of headers do not set any
        additional headers and terminate this set of steps.
        """
    request = webob.Request.blank('/')
    request.method = 'OPTIONS'
    request.headers['Origin'] = 'http://headers.example.com'
    request.headers['Access-Control-Request-Method'] = 'GET'
    request.headers['Access-Control-Request-Headers'] = 'X-Not-Exposed,X-Never-Exposed'
    response = request.get_response(self.application)
    self.assertCORSResponse(response, status='200 OK', allow_origin=None, max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None)