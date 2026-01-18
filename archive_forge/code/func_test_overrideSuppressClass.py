import unittest as pyunit
from twisted.python.reflect import namedAny
from twisted.trial import unittest
from twisted.trial.test import suppression
def test_overrideSuppressClass(self):
    """
        The suppression set on a test method completely overrides a suppression
        with wider scope; if it does not match a warning emitted by that test
        method, the warning is emitted, even if a wider suppression matches.
        """
    self.runTests(self._load(self.TestSuppression, 'testOverrideSuppressClass'))
    warningsShown = self.flushWarnings([self.TestSuppression._emit])
    self.assertEqual(warningsShown[0]['message'], suppression.METHOD_WARNING_MSG)
    self.assertEqual(warningsShown[1]['message'], suppression.CLASS_WARNING_MSG)
    self.assertEqual(warningsShown[2]['message'], suppression.MODULE_WARNING_MSG)
    self.assertEqual(len(warningsShown), 3)