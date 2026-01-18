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
def test_statusWrittenFromThread(self):
    """
        The response status is set on the request object in the reactor thread.
        """
    self.enableThreads()
    invoked = []

    class ThreadVerifier(Request):

        def setResponseCode(self, code, message):
            invoked.append(getThreadID())
            return Request.setResponseCode(self, code, message)

    def applicationFactory():

        def application(environ, startResponse):
            startResponse('200 OK', [])
            return iter(())
        return application
    d, requestFactory = self.requestFactoryFactory(ThreadVerifier)

    def cbRendered(ignored):
        self.assertEqual(set(invoked), {getThreadID()})
    d.addCallback(cbRendered)
    self.lowLevelRender(requestFactory, applicationFactory, DummyChannel, 'GET', '1.1', [], [''])
    return d