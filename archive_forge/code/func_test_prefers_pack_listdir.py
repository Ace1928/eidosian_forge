from dulwich.objects import Blob
from dulwich.tests.test_object_store import PackBasedObjectStoreTests
from dulwich.tests.utils import make_object
from ...tests import TestCaseWithTransport
from ..transportgit import TransportObjectStore, TransportRefsContainer
def test_prefers_pack_listdir(self):
    self.store.add_object(make_object(Blob, data=b'data'))
    self.assertEqual(0, len(self.store.packs))
    self.store.pack_loose_objects()
    self.assertEqual(1, len(self.store.packs), self.store.packs)
    packname = list(self.store.packs)[0].name()
    self.assertEqual({'pack-%s' % packname.decode('ascii')}, set(self.store._pack_names()))
    self.store.transport.put_bytes_non_atomic('info/packs', b'P foo-pack.pack\n')
    self.assertEqual({'pack-%s' % packname.decode('ascii')}, set(self.store._pack_names()))