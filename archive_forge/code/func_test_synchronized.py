from twisted.trial.unittest import TestCase
from twisted.internet.defer import Deferred, fail, succeed
from .._resultstore import ResultStore
from .._eventloop import EventualResult
def test_synchronized(self):
    """
        store() and retrieve() are synchronized.
        """
    self.assertTrue(ResultStore.store.synchronized)
    self.assertTrue(ResultStore.retrieve.synchronized)
    self.assertTrue(ResultStore.log_errors.synchronized)