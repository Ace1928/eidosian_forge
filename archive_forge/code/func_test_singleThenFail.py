import time
from twisted.internet import reactor, task
from twisted.python import failure, log
from twisted.trial import _synctest, reporter, unittest
def test_singleThenFail(self):
    """
            Log a single error, then fail.
            """
    log.err(makeFailure())
    1 + None