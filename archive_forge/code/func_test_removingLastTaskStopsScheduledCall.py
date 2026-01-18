from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def test_removingLastTaskStopsScheduledCall(self):
    """
        If the last task in a Cooperator is removed, the scheduled call for
        the next tick is cancelled, since it is no longer necessary.

        This behavior is useful for tests that want to assert they have left
        no reactor state behind when they're done.
        """
    calls = [None]

    def sched(f):
        calls[0] = FakeDelayedCall(f)
        return calls[0]
    coop = task.Cooperator(scheduler=sched)
    task1 = coop.cooperate(iter([1, 2]))
    task2 = coop.cooperate(iter([1, 2]))
    self.assertEqual(calls[0].func, coop._tick)
    task1.stop()
    self.assertFalse(calls[0].cancelled)
    self.assertEqual(coop._delayedCall, calls[0])
    task2.stop()
    self.assertTrue(calls[0].cancelled)
    self.assertIsNone(coop._delayedCall)
    coop.cooperate(iter([1, 2]))
    self.assertFalse(calls[0].cancelled)
    self.assertEqual(coop._delayedCall, calls[0])