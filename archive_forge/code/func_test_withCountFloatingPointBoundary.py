from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_withCountFloatingPointBoundary(self):
    """
        L{task.LoopingCall.withCount} should never invoke its callable with a
        zero.  Specifically, if a L{task.LoopingCall} created with C{withCount}
        has its L{start <task.LoopingCall.start>} method invoked with a
        floating-point number which introduces decimal inaccuracy when
        multiplied or divided, such as "0.1", L{task.LoopingCall} will never
        invoke its callable with 0.  Also, the sum of all the values passed to
        its callable as the "count" will be an integer, the number of intervals
        that have elapsed.

        This is a regression test for a particularly tricky case to implement.
        """
    clock = task.Clock()
    accumulator = []
    call = task.LoopingCall.withCount(accumulator.append)
    call.clock = clock
    count = 10
    timespan = 1.0
    interval = timespan / count
    call.start(interval, now=False)
    for x in range(count):
        clock.advance(interval)

    def sum_compat(items):
        """
            Make sure the result is more precise.
            On Python 3.11 or older this can be a float with ~ 0.00001
            in precision difference.
            See: https://github.com/python/cpython/issues/100425
            """
        total = 0.0
        for item in items:
            total += item
        return total
    epsilon = timespan - sum_compat([interval] * count)
    clock.advance(epsilon)
    secondsValue = clock.seconds()
    self.assertTrue(abs(epsilon) > 0.0, f'{epsilon} should be greater than zero')
    self.assertTrue(secondsValue >= timespan, f'{secondsValue} should be greater than or equal to {timespan}')
    self.assertEqual(sum_compat(accumulator), count)
    self.assertNotIn(0, accumulator)