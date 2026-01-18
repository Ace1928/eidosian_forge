from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
def test_renderWithQuery(self):
    """
        Test that L{ReverseProxyResource.render} passes query parameters to the
        created factory.
        """
    return self._testRender(b'/index?foo=bar', b'/path?foo=bar')