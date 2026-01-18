from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def test_stopExhausted(self):
    """
        C{stop()}ping a L{CooperativeTask} whose iterator has been exhausted
        should raise L{TaskDone}.
        """
    self.stopNext()
    self.scheduler.pump()
    self.assertRaises(task.TaskDone, self.task.stop)