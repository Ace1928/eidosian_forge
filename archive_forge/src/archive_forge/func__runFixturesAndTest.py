import inspect
import warnings
from typing import Callable, List
from zope.interface import implementer
from typing_extensions import ParamSpec
from twisted.internet import defer, utils
from twisted.python import failure
from twisted.trial import itrial, util
from twisted.trial._synctest import FailTest, SkipTest, SynchronousTestCase
def _runFixturesAndTest(self, result):
    """
        Really run C{setUp}, the test method, and C{tearDown}.  Any of these may
        return L{defer.Deferred}s. After they complete, do some reactor cleanup.

        @param result: A L{TestResult} object.
        """
    from twisted.internet import reactor
    self._deprecateReactor(reactor)
    self._timedOut = False
    try:
        d = self.deferSetUp(None, result)
        try:
            self._wait(d)
        finally:
            self._cleanUp(result)
            self._classCleanUp(result)
    finally:
        self._undeprecateReactor(reactor)