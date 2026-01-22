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
def _remove_pack_from_memory(self, pack):
    """Remove pack from the packs accessed by this repository.

        Only affects memory state, until self._save_pack_names() is invoked.
        """
    self._names.pop(pack.name)
    self._packs_by_name.pop(pack.name)
    self._remove_pack_indices(pack)
    self.packs.remove(pack)