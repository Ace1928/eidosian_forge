from keystone.server.flask.request_processing.middleware import url_normalize
from keystone.tests import unit
def test_rewrite_empty_path(self):
    """Test empty path is rewritten to root."""
    environ = {'PATH_INFO': ''}
    self.middleware(environ, None)
    self.assertEqual('/', self.fake_app.env['PATH_INFO'])
    self.assertEqual(1, len(self.fake_app.env.keys()))