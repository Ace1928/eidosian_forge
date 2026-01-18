from dulwich.objects import Blob
from dulwich.tests.test_object_store import PackBasedObjectStoreTests
from dulwich.tests.utils import make_object
from ...tests import TestCaseWithTransport
from ..transportgit import TransportObjectStore, TransportRefsContainer
def test_remembers_packs(self):
    self.store.add_object(make_object(Blob, data=b'data'))
    self.assertEqual(0, len(self.store.packs))
    self.store.pack_loose_objects()
    self.assertEqual(1, len(self.store.packs))
    self.store.add_object(make_object(Blob, data=b'more data'))
    self.store.pack_loose_objects()
    self.assertEqual(2, len(self.store.packs))
    restore = TransportObjectStore(self.get_transport())
    self.assertEqual(2, len(restore.packs))