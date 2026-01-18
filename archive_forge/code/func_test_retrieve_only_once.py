from twisted.trial.unittest import TestCase
from twisted.internet.defer import Deferred, fail, succeed
from .._resultstore import ResultStore
from .._eventloop import EventualResult
def test_retrieve_only_once(self):
    """
        Once a result is retrieved, it can no longer be retrieved again.
        """
    store = ResultStore()
    dr = EventualResult(Deferred(), None)
    uid = store.store(dr)
    store.retrieve(uid)
    self.assertRaises(KeyError, store.retrieve, uid)