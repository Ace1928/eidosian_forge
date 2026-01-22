from keystone.server.flask.request_processing.middleware import url_normalize
from keystone.tests import unit
class UrlMiddlewareTest(unit.TestCase):

    def setUp(self):
        super(UrlMiddlewareTest, self).setUp()
        self.fake_app = FakeApp()
        self.middleware = url_normalize.URLNormalizingMiddleware(self.fake_app)

    def test_trailing_slash_normalization(self):
        """Test /v3/auth/tokens & /v3/auth/tokens/ normalized URLs match."""
        expected = '/v3/auth/tokens'
        no_slash = {'PATH_INFO': expected}
        with_slash = {'PATH_INFO': '/v3/auth/tokens/'}
        with_many_slash = {'PATH_INFO': '/v3/auth/tokens////'}
        self.middleware(no_slash, None)
        self.assertEqual(expected, self.fake_app.env['PATH_INFO'])
        self.assertEqual(1, len(self.fake_app.env.keys()))
        self.middleware(with_slash, None)
        self.assertEqual(expected, self.fake_app.env['PATH_INFO'])
        self.assertEqual(1, len(self.fake_app.env.keys()))
        self.middleware(with_many_slash, None)
        self.assertEqual(expected, self.fake_app.env['PATH_INFO'])
        self.assertEqual(1, len(self.fake_app.env.keys()))

    def test_rewrite_empty_path(self):
        """Test empty path is rewritten to root."""
        environ = {'PATH_INFO': ''}
        self.middleware(environ, None)
        self.assertEqual('/', self.fake_app.env['PATH_INFO'])
        self.assertEqual(1, len(self.fake_app.env.keys()))