from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
def test_render_subpage(self):
    """
        Test that L{ReverseProxyResource.render} will instantiate a child
        resource that will initiate a connection to the given server
        requesting the apropiate url subpath.
        """
    return self._testRender(b'/index/page1', b'/path/page1')