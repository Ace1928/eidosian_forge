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
def _replace_index_with_readonly(self, index_type):
    unlimited_cache = False
    if index_type == 'chk':
        unlimited_cache = True
    index = self.index_class(self.index_transport, self.index_name(index_type, self.name), self.index_sizes[self.index_offset(index_type)], unlimited_cache=unlimited_cache)
    if index_type == 'chk':
        index._leaf_factory = btree_index._gcchk_factory
    setattr(self, index_type + '_index', index)