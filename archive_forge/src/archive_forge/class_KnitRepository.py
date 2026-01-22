from typing import Type
from ..lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import transport as _mod_transport
from ..repository import InterRepository, IsInWriteGroupError, Repository
from .repository import RepositoryFormatMetaDir
from .serializer import Serializer
from .vf_repository import (InterSameDataRepository,
class KnitRepository(MetaDirVersionedFileRepository):
    """Knit format repository."""
    _commit_builder_class: Type[VersionedFileCommitBuilder]
    _serializer: Serializer

    def __init__(self, _format, a_controldir, control_files, _commit_builder_class, _serializer):
        super().__init__(_format, a_controldir, control_files)
        self._commit_builder_class = _commit_builder_class
        self._serializer = _serializer
        self._reconcile_fixes_text_parents = True

    def _all_revision_ids(self):
        """See Repository.all_revision_ids()."""
        with self.lock_read():
            return [key[0] for key in self.revisions.keys()]

    def _activate_new_inventory(self):
        """Put a replacement inventory.new into use as inventories."""
        t = self._transport
        t.copy('inventory.new.kndx', 'inventory.kndx')
        try:
            t.copy('inventory.new.knit', 'inventory.knit')
        except _mod_transport.NoSuchFile:
            t.delete('inventory.knit')
        t.delete('inventory.new.kndx')
        try:
            t.delete('inventory.new.knit')
        except _mod_transport.NoSuchFile:
            pass
        self.inventories._index._reset_cache()
        self.inventories.keys()

    def _backup_inventory(self):
        t = self._transport
        t.copy('inventory.kndx', 'inventory.backup.kndx')
        t.copy('inventory.knit', 'inventory.backup.knit')

    def _move_file_id(self, from_id, to_id):
        t = self._transport.clone('knits')
        from_rel_url = self.texts._index._mapper.map((from_id, None))
        to_rel_url = self.texts._index._mapper.map((to_id, None))
        for suffix in ('.knit', '.kndx'):
            t.rename(from_rel_url + suffix, to_rel_url + suffix)

    def _remove_file_id(self, file_id):
        t = self._transport.clone('knits')
        rel_url = self.texts._index._mapper.map((file_id, None))
        for suffix in ('.kndx', '.knit'):
            try:
                t.delete(rel_url + suffix)
            except _mod_transport.NoSuchFile:
                pass

    def _temp_inventories(self):
        result = self._format._get_inventories(self._transport, self, 'inventory.new')
        result.get_parent_map([(b'A',)])
        return result

    def get_revision(self, revision_id):
        """Return the Revision object for a named revision"""
        with self.lock_read():
            return self.get_revision_reconcile(revision_id)

    def _refresh_data(self):
        if not self.is_locked():
            return
        if self.is_in_write_group():
            raise IsInWriteGroupError(self)
        self.control_files._finish_transaction()
        if self.is_write_locked():
            self.control_files._set_write_transaction()
        else:
            self.control_files._set_read_transaction()

    def reconcile(self, other=None, thorough=False):
        """Reconcile this repository."""
        from .reconcile import KnitReconciler
        with self.lock_write():
            reconciler = KnitReconciler(self, thorough=thorough)
            return reconciler.reconcile()

    def _make_parents_provider(self):
        return _KnitsParentsProvider(self.revisions)