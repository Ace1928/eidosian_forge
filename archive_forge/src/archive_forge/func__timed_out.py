from fixtures import Fixture
import signal
from typing import Union
from ._deferreddebug import DebugTwisted
from twisted.internet import defer
from twisted.internet.interfaces import IReactorThreads
from twisted.python.failure import Failure
from twisted.python.util import mergeFunctionMetadata
def _timed_out(self, function, timeout):
    e = TimeoutError(function, timeout)
    self._failure = Failure(e)
    self._stop_reactor()