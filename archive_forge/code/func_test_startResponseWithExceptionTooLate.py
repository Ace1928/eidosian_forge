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
def test_startResponseWithExceptionTooLate(self):
    """
        If the I{start_response} callable is invoked with a third positional
        argument after the status and headers have been written to the
        response, the supplied I{exc_info} values are re-raised to the
        application.
        """
    channel = DummyChannel()

    class SomeException(Exception):
        pass
    try:
        raise SomeException()
    except BaseException:
        excInfo = exc_info()
    reraised = []

    def applicationFactory():

        def application(environ, startResponse):
            startResponse('200 OK', [])
            yield b'foo'
            try:
                startResponse('500 ERR', [], excInfo)
            except BaseException:
                reraised.append(exc_info())
        return application
    d, requestFactory = self.requestFactoryFactory()

    def cbRendered(ignored):
        self.assertTrue(channel.transport.written.getvalue().startswith(b'HTTP/1.1 200 OK\r\n'))
        self.assertEqual(reraised[0][0], excInfo[0])
        self.assertEqual(reraised[0][1], excInfo[1])
        tb1 = reraised[0][2].tb_next
        tb2 = excInfo[2]
        self.assertEqual(traceback.extract_tb(tb1)[1], traceback.extract_tb(tb2)[0])
    d.addCallback(cbRendered)
    self.lowLevelRender(requestFactory, applicationFactory, lambda: channel, 'GET', '1.1', [], [''], None, [])
    return d