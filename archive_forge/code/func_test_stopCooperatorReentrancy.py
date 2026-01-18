from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def test_stopCooperatorReentrancy(self):
    """
        If a callback of a L{Deferred} from L{CooperativeTask.whenDone} calls
        C{Cooperator.stop} on its L{CooperativeTask._cooperator}, the
        L{Cooperator} will stop, but the L{CooperativeTask} whose callback is
        calling C{stop} should already be considered 'stopped' by the time the
        callback is running, and therefore removed from the
        L{CoooperativeTask}.
        """
    callbackPhases = []

    def stopit(result):
        callbackPhases.append(result)
        self.cooperator.stop()
        callbackPhases.append('done')
    self.task.whenDone().addCallback(stopit)
    self.stopNext()
    self.scheduler.pump()
    self.assertEqual(callbackPhases, [self.task._iterator, 'done'])