from fixtures import Fixture
import signal
from typing import Union
from ._deferreddebug import DebugTwisted
from twisted.internet import defer
from twisted.internet.interfaces import IReactorThreads
from twisted.python.failure import Failure
from twisted.python.util import mergeFunctionMetadata
def _fake_stop(self):
    """Use to replace ``reactor.stop`` while running a test.

        Calling ``reactor.stop`` makes it impossible re-start the reactor.
        Since the default signal handlers for TERM, BREAK and INT all call
        ``reactor.stop()``, we patch it over with ``reactor.crash()``

        Spinner never calls this method.
        """
    self._reactor.crash()