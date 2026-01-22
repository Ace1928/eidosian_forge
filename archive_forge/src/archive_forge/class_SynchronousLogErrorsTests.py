import time
from twisted.internet import reactor, task
from twisted.python import failure, log
from twisted.trial import _synctest, reporter, unittest
class SynchronousLogErrorsTests(LogErrorsMixin, unittest.SynchronousTestCase):
    MockTest = Mask.SynchronousFailureLogging