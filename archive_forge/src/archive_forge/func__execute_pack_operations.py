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
def _execute_pack_operations(self, pack_operations, packer_class, reload_func=None):
    """Execute a series of pack operations.

        :param pack_operations: A list of [revision_count, packs_to_combine].
        :param packer_class: The class of packer to use
        :return: The new pack names.
        """
    for revision_count, packs in pack_operations:
        if len(packs) == 0:
            continue
        packer = packer_class(self, packs, '.autopack', reload_func=reload_func)
        try:
            result = packer.pack()
        except RetryWithNewPacks:
            if packer.new_pack is not None:
                packer.new_pack.abort()
            raise
        if result is None:
            return
        for pack in packs:
            self._remove_pack_from_memory(pack)
    to_be_obsoleted = []
    for _, packs in pack_operations:
        to_be_obsoleted.extend(packs)
    result = self._save_pack_names(clear_obsolete_packs=True, obsolete_packs=to_be_obsoleted)
    return result