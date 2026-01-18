from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def test_stopStops(self):
    """
        C{stop()}ping a task should cause it to be removed from the run just as
        C{pause()}ing, with the distinction that C{resume()} will raise a
        L{TaskStopped} exception.
        """
    self.task.stop()
    self.scheduler.pump()
    self.assertEqual(len(self.work), 0)
    self.assertRaises(task.TaskStopped, self.task.stop)
    self.assertRaises(task.TaskStopped, self.task.pause)
    self.scheduler.pump()
    self.assertEqual(self.work, [])