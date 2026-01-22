from fixtures import Fixture
import signal
from typing import Union
from ._deferreddebug import DebugTwisted
from twisted.internet import defer
from twisted.internet.interfaces import IReactorThreads
from twisted.python.failure import Failure
from twisted.python.util import mergeFunctionMetadata
def _stop_reactor(self, ignored=None):
    """Stop the reactor!"""
    if self._spinning:
        self._reactor.crash()
        self._spinning = False