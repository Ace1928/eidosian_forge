from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_returnValueNonLocalDeferred(self):
    """
        L{returnValue} will emit a non-local warning in the case where the
        L{inlineCallbacks}-decorated function has already yielded a Deferred
        and therefore moved its generator function along.
        """
    cause = Deferred()

    @inlineCallbacks
    def inline():
        yield cause
        self.mistakenMethod()
        returnValue(2)
    effect = inline()
    results = []
    effect.addCallback(results.append)
    self.assertEqual(results, [])
    cause.callback(1)
    self.assertMistakenMethodWarning(results)