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
def _remove_pack_indices(self, pack, ignore_missing=False):
    """Remove the indices for pack from the aggregated indices.

        :param ignore_missing: Suppress KeyErrors from calling remove_index.
        """
    for index_type in Pack.index_definitions:
        attr_name = index_type + '_index'
        aggregate_index = getattr(self, attr_name)
        if aggregate_index is not None:
            pack_index = getattr(pack, attr_name)
            try:
                aggregate_index.remove_index(pack_index)
            except KeyError:
                if ignore_missing:
                    continue
                raise