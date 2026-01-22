from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
def _eb(result):
    childResultHolder[0] = result.check(CancelledError)
    return result