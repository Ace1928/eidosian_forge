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
class PackCommitBuilder(VersionedFileCommitBuilder):
    """Subclass of VersionedFileCommitBuilder to add texts with pack semantics.

    Specifically this uses one knit object rather than one knit object per
    added text, reducing memory and object pressure.
    """

    def __init__(self, repository, parents, config, timestamp=None, timezone=None, committer=None, revprops=None, revision_id=None, lossy=False, owns_transaction=True):
        VersionedFileCommitBuilder.__init__(self, repository, parents, config, timestamp=timestamp, timezone=timezone, committer=committer, revprops=revprops, revision_id=revision_id, lossy=lossy, owns_transaction=owns_transaction)
        self._file_graph = graph.Graph(repository._pack_collection.text_index.combined_index)

    def _heads(self, file_id, revision_ids):
        keys = [(file_id, revision_id) for revision_id in revision_ids]
        return {key[1] for key in self._file_graph.heads(keys)}