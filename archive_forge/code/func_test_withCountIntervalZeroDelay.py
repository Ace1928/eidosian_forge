from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_withCountIntervalZeroDelay(self):
    """
        L{task.LoopingCall.withCount} with interval set to 0 and a delayed
        call during the loop run will still call the countCallable 1 as if
        no delay occurred.
        """
    clock = task.Clock()
    deferred = defer.Deferred()
    accumulator = []

    def foo(cnt):
        accumulator.append(cnt)
        if len(accumulator) == 2:
            return deferred
        if len(accumulator) > 4:
            loop.stop()
    loop = task.LoopingCall.withCount(foo)
    loop.clock = clock
    loop.start(0, now=False)
    clock.pump([0] * 2)
    self.assertEqual([1] * 2, accumulator)
    clock.pump([1] * 2)
    self.assertEqual([1] * 2, accumulator)
    deferred.callback(None)
    clock.pump([0] * 4)
    self.assertEqual([1] * 5, accumulator)