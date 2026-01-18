from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def test_stopDeferred(self):
    """
        As a corrolary of the interaction of C{pause()} and C{unpause()},
        C{stop()}ping a task which is waiting on a L{Deferred} should cause the
        task to gracefully shut down, meaning that it should not be unpaused
        when the deferred fires.
        """
    self.deferNext()
    self.scheduler.pump()
    d = self.work.pop()
    self.assertEqual(self.task._pauseCount, 1)
    results = []
    d.addBoth(results.append)
    self.scheduler.pump()
    self.task.stop()
    self.scheduler.pump()
    d.callback(7)
    self.scheduler.pump()
    self.assertEqual(results, [None])
    self.assertEqual(self.work, [])