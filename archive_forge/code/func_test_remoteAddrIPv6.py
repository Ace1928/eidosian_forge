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
def test_remoteAddrIPv6(self):
    """
        The C{'REMOTE_ADDR'} key of the C{environ} C{dict} passed to
        the application contains the address of the client making the
        request when connecting over IPv6.
        """

    def channelFactory():
        return DummyChannel(peer=IPv6Address('TCP', '::1', 1234))
    d = self.render('GET', '1.1', [], [''], channelFactory=channelFactory)
    d.addCallback(self.environKeyEqual('REMOTE_ADDR', '::1'))
    return d