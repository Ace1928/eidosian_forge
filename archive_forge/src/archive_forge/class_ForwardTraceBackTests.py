from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
class ForwardTraceBackTests(SynchronousTestCase):

    def test_forwardTracebacks(self):
        """
        Chained inlineCallbacks are forwarding the traceback information
        from generator to generator.

        A first simple test with a couple of inline callbacks.
        """

        @inlineCallbacks
        def erroring():
            yield 'forcing generator'
            raise Exception('Error Marker')

        @inlineCallbacks
        def calling():
            yield erroring()
        d = calling()
        f = self.failureResultOf(d)
        tb = f.getTraceback()
        self.assertIn('in erroring', tb)
        self.assertIn('in calling', tb)
        self.assertIn('Error Marker', tb)

    def test_forwardLotsOfTracebacks(self):
        """
        Several Chained inlineCallbacks gives information about all generators.

        A wider test with a 4 chained inline callbacks.

        Application stack-trace should be reported, and implementation details
        like "throwExceptionIntoGenerator" symbols are omitted from the stack.

        Note that the previous test is testing the simple case, and this one is
        testing the deep recursion case.

        That case needs specific code in failure.py to accomodate to stack
        breakage introduced by throwExceptionIntoGenerator.

        Hence we keep the two tests in order to sort out which code we
        might have regression in.
        """

        @inlineCallbacks
        def erroring():
            yield 'forcing generator'
            raise Exception('Error Marker')

        @inlineCallbacks
        def calling3():
            yield erroring()

        @inlineCallbacks
        def calling2():
            yield calling3()

        @inlineCallbacks
        def calling():
            yield calling2()
        d = calling()
        f = self.failureResultOf(d)
        tb = f.getTraceback()
        self.assertIn('in erroring', tb)
        self.assertIn('in calling', tb)
        self.assertIn('in calling2', tb)
        self.assertIn('in calling3', tb)
        self.assertNotIn('throwExceptionIntoGenerator', tb)
        self.assertIn('Error Marker', tb)
        self.assertIn('in erroring', f.getTraceback())