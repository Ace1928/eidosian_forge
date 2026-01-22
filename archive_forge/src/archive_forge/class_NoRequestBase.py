import webob
from oslo_middleware.base import ConfigurableMiddleware
from oslo_middleware.base import Middleware
from oslotest.base import BaseTestCase
class NoRequestBase(Middleware):
    """Test middleware, implements old model."""

    def process_response(self, response):
        self.called_without_request = True
        return response