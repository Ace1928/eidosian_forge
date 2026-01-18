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
def test_scriptName(self):
    """
        The C{'SCRIPT_NAME'} key of the C{environ} C{dict} passed to the
        application contains the I{abs_path} (RFC 2396, section 3) to this
        resource (RFC 3875, section 4.1.13).
        """
    root = self.render('GET', '1.1', [], [''])
    root.addCallback(self.environKeyEqual('SCRIPT_NAME', ''))
    emptyChild = self.render('GET', '1.1', [''], [''])
    emptyChild.addCallback(self.environKeyEqual('SCRIPT_NAME', '/'))
    leaf = self.render('GET', '1.1', ['foo'], ['foo'])
    leaf.addCallback(self.environKeyEqual('SCRIPT_NAME', '/foo'))
    container = self.render('GET', '1.1', ['foo', ''], ['foo', ''])
    container.addCallback(self.environKeyEqual('SCRIPT_NAME', '/foo/'))
    internal = self.render('GET', '1.1', ['foo'], ['foo', 'bar'])
    internal.addCallback(self.environKeyEqual('SCRIPT_NAME', '/foo'))
    unencoded = self.render('GET', '1.1', ['foo', '/', b'bar\xff'], ['foo', '/', b'bar\xff'])
    unencoded.addCallback(self.environKeyEqual('SCRIPT_NAME', '/foo///barÿ'))
    return gatherResults([root, emptyChild, leaf, container, internal, unencoded])