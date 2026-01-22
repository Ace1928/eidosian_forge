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
def _connect_deferred(self, deferred):
    """
        Hook up the Deferred that that this will be the result of.

        Should only be run in Twisted thread, and only called once.
        """
    self._deferred = deferred

    def put(result, eventual=weakref.ref(self)):
        eventual = eventual()
        if eventual:
            eventual._set_result(result)
        else:
            err(result, 'Unhandled error in EventualResult')
    deferred.addBoth(put)