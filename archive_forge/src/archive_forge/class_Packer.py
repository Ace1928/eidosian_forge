import re
import sys
from typing import Type
from ..lazy_import import lazy_import
import contextlib
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.index import (
from .. import errors, lockable_files, lockdir
from .. import transport as _mod_transport
from ..bzr import btree_index, index
from ..decorators import only_raises
from ..lock import LogicalLockResult
from ..repository import RepositoryWriteLockResult, _LazyListJoin
from ..trace import mutter, note, warning
from .repository import MetaDirRepository, RepositoryFormatMetaDir
from .serializer import Serializer
from .vf_repository import (MetaDirVersionedFileRepository,
class Packer:
    """Create a pack from packs."""

    def __init__(self, pack_collection, packs, suffix, revision_ids=None, reload_func=None):
        """Create a Packer.

        :param pack_collection: A RepositoryPackCollection object where the
            new pack is being written to.
        :param packs: The packs to combine.
        :param suffix: The suffix to use on the temporary files for the pack.
        :param revision_ids: Revision ids to limit the pack to.
        :param reload_func: A function to call if a pack file/index goes
            missing. The side effect of calling this function should be to
            update self.packs. See also AggregateIndex
        """
        self.packs = packs
        self.suffix = suffix
        self.revision_ids = revision_ids
        self.new_pack = None
        self._pack_collection = pack_collection
        self._reload_func = reload_func
        self._revision_keys = None
        self._text_filter = None

    def pack(self, pb=None):
        """Create a new pack by reading data from other packs.

        This does little more than a bulk copy of data. One key difference
        is that data with the same item key across multiple packs is elided
        from the output. The new pack is written into the current pack store
        along with its indices, and the name added to the pack names. The
        source packs are not altered and are not required to be in the current
        pack collection.

        :param pb: An optional progress bar to use. A nested bar is created if
            this is None.
        :return: A Pack object, or None if nothing was copied.
        """
        if self._pack_collection._new_pack is not None:
            raise errors.BzrError('call to %s.pack() while another pack is being written.' % (self.__class__.__name__,))
        if self.revision_ids is not None:
            if len(self.revision_ids) == 0:
                return None
            else:
                self.revision_ids = frozenset(self.revision_ids)
                self.revision_keys = frozenset(((revid,) for revid in self.revision_ids))
        if pb is None:
            self.pb = ui.ui_factory.nested_progress_bar()
        else:
            self.pb = pb
        try:
            return self._create_pack_from_packs()
        finally:
            if pb is None:
                self.pb.finished()

    def open_pack(self):
        """Open a pack for the pack we are creating."""
        new_pack = self._pack_collection.pack_factory(self._pack_collection, upload_suffix=self.suffix, file_mode=self._pack_collection.repo.controldir._get_file_mode())
        new_pack.revision_index.set_optimize(combine_backing_indices=False)
        new_pack.inventory_index.set_optimize(combine_backing_indices=False)
        new_pack.text_index.set_optimize(combine_backing_indices=False)
        new_pack.signature_index.set_optimize(combine_backing_indices=False)
        return new_pack

    def _copy_revision_texts(self):
        """Copy revision data to the new pack."""
        raise NotImplementedError(self._copy_revision_texts)

    def _copy_inventory_texts(self):
        """Copy the inventory texts to the new pack.

        self._revision_keys is used to determine what inventories to copy.

        Sets self._text_filter appropriately.
        """
        raise NotImplementedError(self._copy_inventory_texts)

    def _copy_text_texts(self):
        raise NotImplementedError(self._copy_text_texts)

    def _create_pack_from_packs(self):
        raise NotImplementedError(self._create_pack_from_packs)

    def _log_copied_texts(self):
        if 'pack' in debug.debug_flags:
            mutter('%s: create_pack: file texts copied: %s%s %d items t+%6.3fs', time.ctime(), self._pack_collection._upload_transport.base, self.new_pack.random_name, self.new_pack.text_index.key_count(), time.time() - self.new_pack.start_time)

    def _use_pack(self, new_pack):
        """Return True if new_pack should be used.

        :param new_pack: The pack that has just been created.
        :return: True if the pack should be used.
        """
        return new_pack.data_inserted()