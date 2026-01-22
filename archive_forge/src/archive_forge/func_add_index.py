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
def add_index(self, index, pack):
    """Add index to the aggregate, which is an index for Pack pack.

        Future searches on the aggregate index will seach this new index
        before all previously inserted indices.

        :param index: An Index for the pack.
        :param pack: A Pack instance.
        """
    self.index_to_pack[index] = pack.access_tuple()
    self.combined_index.insert_index(0, index, pack.name)