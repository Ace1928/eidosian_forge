from .. import errors
from .. import transport as _mod_transport
from ..lazy_import import lazy_import
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.knit import (
from ..bzr import btree_index
from ..bzr.index import (CombinedGraphIndex, GraphIndex,
from ..bzr.vf_repository import StreamSource
from .knitrepo import KnitRepository
from .pack_repo import (NewPack, PackCommitBuilder, Packer, PackRepository,
class KnitPackRepository(PackRepository, KnitRepository):

    def __init__(self, _format, a_controldir, control_files, _commit_builder_class, _serializer):
        PackRepository.__init__(self, _format, a_controldir, control_files, _commit_builder_class, _serializer)
        if self._format.supports_chks:
            raise AssertionError('chk not supported')
        index_transport = self._transport.clone('indices')
        self._pack_collection = KnitRepositoryPackCollection(self, self._transport, index_transport, self._transport.clone('upload'), self._transport.clone('packs'), _format.index_builder_class, _format.index_class, use_chk_index=False)
        self.inventories = KnitVersionedFiles(_KnitGraphIndex(self._pack_collection.inventory_index.combined_index, add_callback=self._pack_collection.inventory_index.add_callback, deltas=True, parents=True, is_locked=self.is_locked), data_access=self._pack_collection.inventory_index.data_access, max_delta_chain=200)
        self.revisions = KnitVersionedFiles(_KnitGraphIndex(self._pack_collection.revision_index.combined_index, add_callback=self._pack_collection.revision_index.add_callback, deltas=False, parents=True, is_locked=self.is_locked, track_external_parent_refs=True), data_access=self._pack_collection.revision_index.data_access, max_delta_chain=0)
        self.signatures = KnitVersionedFiles(_KnitGraphIndex(self._pack_collection.signature_index.combined_index, add_callback=self._pack_collection.signature_index.add_callback, deltas=False, parents=False, is_locked=self.is_locked), data_access=self._pack_collection.signature_index.data_access, max_delta_chain=0)
        self.texts = KnitVersionedFiles(_KnitGraphIndex(self._pack_collection.text_index.combined_index, add_callback=self._pack_collection.text_index.add_callback, deltas=True, parents=True, is_locked=self.is_locked), data_access=self._pack_collection.text_index.data_access, max_delta_chain=200)
        self.chk_bytes = None
        self._write_lock_count = 0
        self._transaction = None
        self._reconcile_does_inventory_gc = True
        self._reconcile_fixes_text_parents = True
        self._reconcile_backsup_inventory = False

    def _get_source(self, to_format):
        if to_format.network_name() == self._format.network_name():
            return KnitPackStreamSource(self, to_format)
        return PackRepository._get_source(self, to_format)

    def _reconcile_pack(self, collection, packs, extension, revs, pb):
        packer = KnitReconcilePacker(collection, packs, extension, revs)
        return packer.pack(pb)