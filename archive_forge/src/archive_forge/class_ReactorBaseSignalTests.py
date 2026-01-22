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
@skipIf(not signal, 'signal module not available')
class ReactorBaseSignalTests(TestCase):
    """
    Tests to exercise ReactorBase's signal exit reporting path.
    """

    def test_exitSignalDefaultsToNone(self):
        """
        The default value of the _exitSignal attribute is None.
        """
        reactor = TestSpySignalCapturingReactor()
        self.assertIs(None, reactor._exitSignal)

    def test_captureSIGINT(self):
        """
        ReactorBase's SIGINT handler saves the value of SIGINT to the
        _exitSignal attribute.
        """
        reactor = TestSpySignalCapturingReactor()
        reactor.sigInt(signal.SIGINT, None)
        self.assertEquals(signal.SIGINT, reactor._exitSignal)

    def test_captureSIGTERM(self):
        """
        ReactorBase's SIGTERM handler saves the value of SIGTERM to the
        _exitSignal attribute.
        """
        reactor = TestSpySignalCapturingReactor()
        reactor.sigTerm(signal.SIGTERM, None)
        self.assertEquals(signal.SIGTERM, reactor._exitSignal)

    def test_captureSIGBREAK(self):
        """
        ReactorBase's SIGBREAK handler saves the value of SIGBREAK to the
        _exitSignal attribute.
        """
        if not hasattr(signal, 'SIGBREAK'):
            raise SkipTest('signal module does not have SIGBREAK')
        reactor = TestSpySignalCapturingReactor()
        reactor.sigBreak(signal.SIGBREAK, None)
        self.assertEquals(signal.SIGBREAK, reactor._exitSignal)