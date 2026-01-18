from fixtures import Fixture
import signal
from typing import Union
from ._deferreddebug import DebugTwisted
from twisted.internet import defer
from twisted.internet.interfaces import IReactorThreads
from twisted.python.failure import Failure
from twisted.python.util import mergeFunctionMetadata
def trap_unhandled_errors(function, *args, **kwargs):
    """Run a function, trapping any unhandled errors in Deferreds.

    Assumes that 'function' will have handled any errors in Deferreds by the
    time it is complete.  This is almost never true of any Twisted code, since
    you can never tell when someone has added an errback to a Deferred.

    If 'function' raises, then don't bother doing any unhandled error
    jiggery-pokery, since something horrible has probably happened anyway.

    :return: A tuple of '(result, error)', where 'result' is the value
        returned by 'function' and 'error' is a list of 'defer.DebugInfo'
        objects that have unhandled errors in Deferreds.
    """
    real_DebugInfo = defer.DebugInfo
    debug_infos = []

    class DebugInfo(real_DebugInfo):
        _runRealDel = True

        def __init__(self):
            real_DebugInfo.__init__(self)
            debug_infos.append(self)

        def __del__(self):
            if self._runRealDel:
                real_DebugInfo.__del__(self)
    defer.DebugInfo = DebugInfo
    try:
        result = function(*args, **kwargs)
    finally:
        defer.DebugInfo = real_DebugInfo
    errors = []
    for info in debug_infos:
        if info.failResult is not None:
            errors.append(info)
            info._runRealDel = False
    return (result, errors)