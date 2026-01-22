from os.path import abspath
from xml.dom.minidom import parseString
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.internet import defer, reactor
from twisted.logger import globalLogPublisher
from twisted.python import failure, filepath
from twisted.spread import pb
from twisted.spread.banana import SIZE_LIMIT
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
from twisted.web import client, distrib, resource, server, static
from twisted.web.http_headers import Headers
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
def _requestTest(self, child, **kwargs):
    """
        Set up a resource on a distrib site using L{ResourcePublisher} and
        then retrieve it from a L{ResourceSubscription} via an HTTP client.

        @param child: The resource to publish using distrib.
        @param **kwargs: Extra keyword arguments to pass to L{Agent.request} when
            requesting the resource.

        @return: A L{Deferred} which fires with the result of the request.
        """
    mainPort, mainAddr = self._setupDistribServer(child)
    agent = client.Agent(reactor)
    url = f'http://{mainAddr.host}:{mainAddr.port}/child'
    url = url.encode('ascii')
    d = agent.request(b'GET', url, **kwargs)
    d.addCallback(client.readBody)
    return d