import threading
import weakref
import warnings
from inspect import iscoroutinefunction
from functools import wraps
from queue import SimpleQueue
from twisted.python import threadable
from twisted.python.runtime import platform
from twisted.python.failure import Failure
from twisted.python.log import PythonLoggingObserver, err
from twisted.internet.defer import maybeDeferred, ensureDeferred
from twisted.internet.task import LoopingCall
import wrapt
from ._util import synchronized
from ._resultstore import ResultStore
def _common_setup(self):
    """
        The minimal amount of setup done by both setup() and no_setup().
        """
    self._started = True
    self._reactor = self._reactorFactory()
    self._registry = ResultRegistry()
    self._reactor.addSystemEventTrigger('before', 'shutdown', self._registry.stop)