import time
from twisted.internet import reactor, task
from twisted.python import failure, log
from twisted.trial import _synctest, reporter, unittest
def test_singleError(self):
    """
        Test that a logged error gets reported as a test error.
        """
    test = self.MockTest('test_single')
    test(self.result)
    self.assertEqual(len(self.result.errors), 1)
    self.assertTrue(self.result.errors[0][1].check(ZeroDivisionError), self.result.errors[0][1])
    self.assertEqual(0, self.result.successes)