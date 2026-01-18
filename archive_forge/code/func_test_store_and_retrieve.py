from twisted.trial.unittest import TestCase
from twisted.internet.defer import Deferred, fail, succeed
from .._resultstore import ResultStore
from .._eventloop import EventualResult
def test_store_and_retrieve(self):
    """
        EventualResult instances be be stored in a ResultStore and then
        retrieved using the id returned from store().
        """
    store = ResultStore()
    dr = EventualResult(Deferred(), None)
    uid = store.store(dr)
    self.assertIdentical(store.retrieve(uid), dr)