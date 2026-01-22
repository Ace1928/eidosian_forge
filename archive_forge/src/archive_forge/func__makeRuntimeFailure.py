import time
from twisted.internet import reactor, task
from twisted.python import failure, log
from twisted.trial import _synctest, reporter, unittest
def _makeRuntimeFailure(self):
    return failure.Failure(RuntimeError('test error'))