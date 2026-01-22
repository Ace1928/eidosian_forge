from fixtures import Fixture
import signal
from typing import Union
from ._deferreddebug import DebugTwisted
from twisted.internet import defer
from twisted.internet.interfaces import IReactorThreads
from twisted.python.failure import Failure
from twisted.python.util import mergeFunctionMetadata
def _save_signals(self):
    available_signals = [getattr(signal, name, None) for name in self._PRESERVED_SIGNALS]
    self._saved_signals = [(sig, signal.getsignal(sig)) for sig in available_signals if sig]