from unittest import skipIf
from twisted.internet.error import ConnectionDone
from twisted.internet.posixbase import _ContinuousPolling
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def test_removeReader(self):
    """
        Removing a reader stops the C{LoopingCall}.
        """
    poller = _ContinuousPolling(Clock())
    reader = object()
    poller.addReader(reader)
    poller.removeReader(reader)
    self.assertIsNone(poller._loop)
    self.assertEqual(poller._reactor.getDelayedCalls(), [])
    self.assertFalse(poller.isReading(reader))