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
def _resume_write_group(self, tokens):
    self._start_write_group()
    try:
        self._pack_collection._resume_write_group(tokens)
    except errors.UnresumableWriteGroup:
        self._abort_write_group()
        raise
    for pack in self._pack_collection._resumed_packs:
        self.revisions._index.scan_unvalidated_index(pack.revision_index)