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
def test_readlinesMoreThan(self):
    """
        Calling L{_InputStream.readlines} with an integer which is greater than
        the total number of bytes in the input stream returns a list of all
        lines from the input.
        """
    bytes = b'one potato\ntwo potato\nthree potato'
    d = self._renderAndReturnReaderResult(lambda input: input.readlines(100), bytes)
    d.addCallback(self.assertEqual, [b'one potato\n', b'two potato\n', b'three potato'])
    return d