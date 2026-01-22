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
class SynchronousReactorThreads:
    """
    A single-threaded implementation of part of the L{IReactorThreads}
    interface.  This implementation assumes that it will only be invoked
    from the reactor thread, so it calls functions synchronously rather than
    trying to schedule them to run in the reactor thread.  It is used in
    conjunction with L{SynchronousThreadPool} to make the tests which are
    not directly for thread-related behavior deterministic.
    """

    def callFromThread(self, f, *a, **kw):
        """
        Call C{f(*a, **kw)} in this thread which should also be the reactor
        thread.
        """
        f(*a, **kw)