from collections import defaultdict
from socket import (
from threading import Lock, local
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted._threads import LockWorker, Team, createMemoryWorker
from twisted.internet._resolver import (
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.base import PluggableResolverMixin, ReactorBase
from twisted.internet.defer import Deferred
from twisted.internet.error import DNSLookupError
from twisted.internet.interfaces import (
from twisted.python.threadpool import ThreadPool
from twisted.trial.unittest import SynchronousTestCase as UnitTest
class DeterministicThreadPool(ThreadPool):
    """
    Create a deterministic L{ThreadPool} object.
    """

    def __init__(self, team):
        """
        Create a L{DeterministicThreadPool} from a L{Team}.
        """
        self.min = 1
        self.max = 1
        self.name = None
        self.threads = []
        self._team = team