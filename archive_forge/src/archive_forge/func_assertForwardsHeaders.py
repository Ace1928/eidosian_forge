from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
def assertForwardsHeaders(self, proxyClient, requestLine, headers):
    """
        Assert that C{proxyClient} sends C{headers} when it connects.

        @param proxyClient: A L{ProxyClient}.
        @param requestLine: The request line we expect to be sent.
        @param headers: A dict of headers we expect to be sent.
        @return: If the assertion is successful, return the request body as
            bytes.
        """
    self.connectProxy(proxyClient)
    requestContent = proxyClient.transport.value()
    receivedLine, receivedHeaders, body = self._parseOutHeaders(requestContent)
    self.assertEqual(receivedLine, requestLine)
    self.assertEqual(receivedHeaders, headers)
    return body