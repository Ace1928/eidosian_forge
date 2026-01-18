from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
def test_processWithData(self):
    """
        L{ProxyRequest.process} should be able to retrieve request body and
        to forward it.
        """
    return self._testProcess(b'/foo/bar', b'/foo/bar', b'POST', b'Some content')