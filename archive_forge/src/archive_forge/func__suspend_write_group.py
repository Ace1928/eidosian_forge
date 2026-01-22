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
def _suspend_write_group(self):
    tokens = [pack.name for pack in self._resumed_packs]
    self._remove_pack_indices(self._new_pack)
    if self._new_pack.data_inserted():
        self._new_pack.finish(suspend=True)
        tokens.append(self._new_pack.name)
        self._new_pack = None
    else:
        self._new_pack.abort()
        self._new_pack = None
    self._remove_resumed_pack_indices()
    return tokens