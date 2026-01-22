from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
def assertMistakenMethodWarning(self, resultList):
    """
        Flush the current warnings and assert that we have been told that
        C{mistakenMethod} was invoked, and that the result from the Deferred
        that was fired (appended to the given list) is C{mistakenMethod}'s
        result.  The warning should indicate that an inlineCallbacks function
        called 'inline' was made to exit.
        """
    self.assertEqual(resultList, [1])
    warnings = self.flushWarnings(offendingFunctions=[self.mistakenMethod])
    self.assertEqual(len(warnings), 1)
    self.assertEqual(warnings[0]['category'], DeprecationWarning)
    self.assertEqual(warnings[0]['message'], "returnValue() in 'mistakenMethod' causing 'inline' to exit: returnValue should only be invoked by functions decorated with inlineCallbacks")