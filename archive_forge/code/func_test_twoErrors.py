import time
from twisted.internet import reactor, task
from twisted.python import failure, log
from twisted.trial import _synctest, reporter, unittest
def test_twoErrors(self):
    """
        Test that when two errors get logged, they both get reported as test
        errors.
        """
    test = self.MockTest('test_double')
    test(self.result)
    self.assertEqual(len(self.result.errors), 2)
    self.assertEqual(0, self.result.successes)