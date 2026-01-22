from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
class CORSTestWildcard(CORSTestBase):
    """Test the CORS wildcard specification."""

    def setUp(self):
        super(CORSTestWildcard, self).setUp()
        fixture = self.config_fixture
        fixture.load_raw_values(group='cors', allowed_origin='http://default.example.com', allow_credentials='True', max_age='', expose_headers='', allow_methods='GET,PUT,POST,DELETE,HEAD', allow_headers='')
        fixture.load_raw_values(group='cors.wildcard', allowed_origin='*', allow_methods='GET')
        self.application = cors.CORS(test_application, self.config)

    def test_config_overrides(self):
        """Assert that the configuration options are properly registered."""
        gc = self.config.cors
        self.assertEqual(['http://default.example.com'], gc.allowed_origin)
        self.assertEqual(True, gc.allow_credentials)
        self.assertEqual([], gc.expose_headers)
        self.assertIsNone(gc.max_age)
        self.assertEqual(['GET', 'PUT', 'POST', 'DELETE', 'HEAD'], gc.allow_methods)
        self.assertEqual([], gc.allow_headers)
        ac = self.config['cors.wildcard']
        self.assertEqual(['*'], ac.allowed_origin)
        self.assertEqual(True, gc.allow_credentials)
        self.assertEqual(gc.expose_headers, ac.expose_headers)
        self.assertEqual(gc.max_age, ac.max_age)
        self.assertEqual(['GET'], ac.allow_methods)
        self.assertEqual(gc.allow_headers, ac.allow_headers)

    def test_wildcard_domain(self):
        """CORS Specification, Wildcards

        If the configuration file specifies CORS settings for the wildcard '*'
        domain, it should return those for all origin domains except for the
        overrides.
        """
        request = webob.Request.blank('/')
        request.method = 'OPTIONS'
        request.headers['Origin'] = 'http://default.example.com'
        request.headers['Access-Control-Request-Method'] = 'GET'
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin='http://default.example.com', max_age=None, allow_methods='GET', allow_headers='', allow_credentials='true', expose_headers=None)
        request = webob.Request.blank('/')
        request.method = 'GET'
        request.headers['Origin'] = 'http://default.example.com'
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin='http://default.example.com', max_age=None, allow_headers='', allow_credentials='true', expose_headers=None, has_content_type=True)
        request = webob.Request.blank('/')
        request.method = 'OPTIONS'
        request.headers['Origin'] = 'http://invalid.example.com'
        request.headers['Access-Control-Request-Method'] = 'GET'
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin='*', max_age=None, allow_methods='GET', allow_headers='', allow_credentials='true', expose_headers=None, has_content_type=True)