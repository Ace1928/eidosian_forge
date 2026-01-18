import webob
from oslo_middleware.base import ConfigurableMiddleware
from oslo_middleware.base import Middleware
from oslotest.base import BaseTestCase
def test_paste_deploy_legacy(self):
    app = LegacyMiddlewareTest.factory({'global': True}, local=True)(application)
    self.assertEqual({}, app.conf)