import tempfile
import traceback
import warnings
from sys import exc_info
from urllib.parse import quote as urlquote
from zope.interface.verify import verifyObject
from twisted.internet import reactor
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.error import ConnectionLost
from twisted.internet.testing import EventLoggingObserver
from twisted.logger import Logger, globalLogPublisher
from twisted.python.failure import Failure
from twisted.python.threadable import getThreadID
from twisted.python.threadpool import ThreadPool
from twisted.trial.unittest import TestCase
from twisted.web import http
from twisted.web.resource import IResource, Resource
from twisted.web.server import Request, Site, version
from twisted.web.test.test_web import DummyChannel
from twisted.web.wsgi import WSGIResource
def _headersTest(self, appHeaders, expectedHeaders):
    """
        Verify that if the response headers given by C{appHeaders} are passed
        to the I{start_response} callable, then the response header lines given
        by C{expectedHeaders} plus I{Server} and I{Date} header lines are
        included in the response.
        """
    self.patch(http, 'datetimeToString', lambda: 'Tuesday')
    channel = DummyChannel()

    def applicationFactory():

        def application(environ, startResponse):
            startResponse('200 OK', appHeaders)
            return iter(())
        return application
    d, requestFactory = self.requestFactoryFactory()

    def cbRendered(ignored):
        response = channel.transport.written.getvalue()
        headers, rest = response.split(b'\r\n\r\n', 1)
        headerLines = headers.split(b'\r\n')[1:]
        headerLines.sort()
        allExpectedHeaders = expectedHeaders + [b'Date: Tuesday', b'Server: ' + version, b'Transfer-Encoding: chunked']
        allExpectedHeaders.sort()
        self.assertEqual(headerLines, allExpectedHeaders)
    d.addCallback(cbRendered)
    self.lowLevelRender(requestFactory, applicationFactory, lambda: channel, 'GET', '1.1', [], [''], None, [])
    return d