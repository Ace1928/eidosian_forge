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
def _renderAndReturnReaderResult(self, reader, content):
    contentType = self.getFileType()

    class CustomizedRequest(Request):

        def gotLength(self, length):
            self.content = contentType()

    def appFactoryFactory(reader):
        result = Deferred()

        def applicationFactory():

            def application(*args):
                environ, startResponse = args
                result.callback(reader(environ['wsgi.input']))
                startResponse('200 OK', [])
                return iter(())
            return application
        return (result, applicationFactory)
    d, appFactory = appFactoryFactory(reader)
    self.lowLevelRender(CustomizedRequest, appFactory, DummyChannel, 'PUT', '1.1', [], [''], None, [], content)
    return d