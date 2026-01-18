import socket
from queue import Queue
from typing import Callable
from unittest import skipIf
from zope.interface import implementer
from typing_extensions import ParamSpec
from twisted.internet._resolver import FirstOneWins
from twisted.internet.base import DelayedCall, ReactorBase, ThreadedResolver
from twisted.internet.defer import Deferred
from twisted.internet.error import DNSLookupError
from twisted.internet.interfaces import IReactorThreads, IReactorTime, IResolverSimple
from twisted.internet.task import Clock
from twisted.python.threadpool import ThreadPool
from twisted.trial.unittest import SkipTest, TestCase
def test_switchToDebug(self):
    """
        If L{DelayedCall.debug} changes from C{0} to C{1} between
        L{DelayeCall.__init__} and L{DelayedCall.__repr__} then
        L{DelayedCall.__repr__} returns a string that does not include the
        creator stack.
        """
    dc = DelayedCall(3, lambda: None, (), {}, nothing, nothing, lambda: 2)
    dc.debug = 1
    self.assertNotIn('traceback at creation', repr(dc))