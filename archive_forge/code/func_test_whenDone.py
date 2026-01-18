from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def test_whenDone(self):
    """
        L{CooperativeTask.whenDone} returns a Deferred which fires when the
        Cooperator's iterator is exhausted.  It returns a new Deferred each
        time it is called; callbacks added to other invocations will not modify
        the value that subsequent invocations will fire with.
        """
    deferred1 = self.task.whenDone()
    deferred2 = self.task.whenDone()
    results1 = []
    results2 = []
    final1 = []
    final2 = []

    def callbackOne(result):
        results1.append(result)
        return 1

    def callbackTwo(result):
        results2.append(result)
        return 2
    deferred1.addCallback(callbackOne)
    deferred2.addCallback(callbackTwo)
    deferred1.addCallback(final1.append)
    deferred2.addCallback(final2.append)
    self.stopNext()
    self.scheduler.pump()
    self.assertEqual(len(results1), 1)
    self.assertEqual(len(results2), 1)
    self.assertIs(results1[0], self.task._iterator)
    self.assertIs(results2[0], self.task._iterator)
    self.assertEqual(final1, [1])
    self.assertEqual(final2, [2])