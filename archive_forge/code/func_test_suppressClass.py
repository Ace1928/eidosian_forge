import unittest as pyunit
from twisted.python.reflect import namedAny
from twisted.trial import unittest
from twisted.trial.test import suppression
def test_suppressClass(self):
    """
        A suppression set on a L{SynchronousTestCase} subclass prevents warnings
        emitted by any test methods defined on that class which match the
        suppression from being emitted.
        """
    self.runTests(self._load(self.TestSuppression, 'testSuppressClass'))
    warningsShown = self.flushWarnings([self.TestSuppression._emit])
    self.assertEqual(warningsShown[0]['message'], suppression.METHOD_WARNING_MSG)
    self.assertEqual(warningsShown[1]['message'], suppression.MODULE_WARNING_MSG)
    self.assertEqual(len(warningsShown), 2)