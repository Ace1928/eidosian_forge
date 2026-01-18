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
def test_serverName(self):
    """
        The C{'SERVER_NAME'} key of the C{environ} C{dict} passed to the
        application contains the best determination of the server hostname
        possible, using either the value of the I{Host} header in the request
        or the address the server is listening on if that header is not
        present (RFC 3875, section 4.1.14).
        """
    missing = self.render('GET', '1.1', [], [''])
    missing.addCallback(self.environKeyEqual('SERVER_NAME', '10.0.0.1'))
    present = self.render('GET', '1.1', [], [''], None, [('host', 'example.org')])
    present.addCallback(self.environKeyEqual('SERVER_NAME', 'example.org'))
    return gatherResults([missing, present])