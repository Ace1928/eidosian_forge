from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def test_runningWhenStarted(self):
    """
        L{Cooperator.running} reports C{True} if the L{Cooperator}
        was started on creation.
        """
    c = task.Cooperator()
    self.assertTrue(c.running)