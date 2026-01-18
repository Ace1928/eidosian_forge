from dulwich.objects import Blob
from dulwich.tests.test_object_store import PackBasedObjectStoreTests
from dulwich.tests.utils import make_object
from ...tests import TestCaseWithTransport
from ..transportgit import TransportObjectStore, TransportRefsContainer
def test_packed_refs_missing(self):
    self.assertEqual({}, self._refs.get_packed_refs())