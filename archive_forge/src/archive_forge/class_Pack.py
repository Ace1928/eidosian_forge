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
class Pack:
    """An in memory proxy for a pack and its indices.

    This is a base class that is not directly used, instead the classes
    ExistingPack and NewPack are used.
    """
    index_definitions = {'chk': ('.cix', 4), 'revision': ('.rix', 0), 'inventory': ('.iix', 1), 'text': ('.tix', 2), 'signature': ('.six', 3)}

    def __init__(self, revision_index, inventory_index, text_index, signature_index, chk_index=None):
        """Create a pack instance.

        :param revision_index: A GraphIndex for determining what revisions are
            present in the Pack and accessing the locations of their texts.
        :param inventory_index: A GraphIndex for determining what inventories are
            present in the Pack and accessing the locations of their
            texts/deltas.
        :param text_index: A GraphIndex for determining what file texts
            are present in the pack and accessing the locations of their
            texts/deltas (via (fileid, revisionid) tuples).
        :param signature_index: A GraphIndex for determining what signatures are
            present in the Pack and accessing the locations of their texts.
        :param chk_index: A GraphIndex for accessing content by CHK, if the
            pack has one.
        """
        self.revision_index = revision_index
        self.inventory_index = inventory_index
        self.text_index = text_index
        self.signature_index = signature_index
        self.chk_index = chk_index

    def access_tuple(self):
        """Return a tuple (transport, name) for the pack content."""
        return (self.pack_transport, self.file_name())

    def _check_references(self):
        """Make sure our external references are present.

        Packs are allowed to have deltas whose base is not in the pack, but it
        must be present somewhere in this collection.  It is not allowed to
        have deltas based on a fallback repository.
        (See <https://bugs.launchpad.net/bzr/+bug/288751>)
        """
        missing_items = {}
        for index_name, external_refs, index in [('texts', self._get_external_refs(self.text_index), self._pack_collection.text_index.combined_index), ('inventories', self._get_external_refs(self.inventory_index), self._pack_collection.inventory_index.combined_index)]:
            missing = external_refs.difference((k for idx, k, v, r in index.iter_entries(external_refs)))
            if missing:
                missing_items[index_name] = sorted(list(missing))
        if missing_items:
            from pprint import pformat
            raise errors.BzrCheckError('Newly created pack file %r has delta references to items not in its repository:\n%s' % (self, pformat(missing_items)))

    def file_name(self):
        """Get the file name for the pack on disk."""
        return self.name + '.pack'

    def get_revision_count(self):
        return self.revision_index.key_count()

    def index_name(self, index_type, name):
        """Get the disk name of an index type for pack name 'name'."""
        return name + Pack.index_definitions[index_type][0]

    def index_offset(self, index_type):
        """Get the position in a index_size array for a given index type."""
        return Pack.index_definitions[index_type][1]

    def inventory_index_name(self, name):
        """The inv index is the name + .iix."""
        return self.index_name('inventory', name)

    def revision_index_name(self, name):
        """The revision index is the name + .rix."""
        return self.index_name('revision', name)

    def signature_index_name(self, name):
        """The signature index is the name + .six."""
        return self.index_name('signature', name)

    def text_index_name(self, name):
        """The text index is the name + .tix."""
        return self.index_name('text', name)

    def _replace_index_with_readonly(self, index_type):
        unlimited_cache = False
        if index_type == 'chk':
            unlimited_cache = True
        index = self.index_class(self.index_transport, self.index_name(index_type, self.name), self.index_sizes[self.index_offset(index_type)], unlimited_cache=unlimited_cache)
        if index_type == 'chk':
            index._leaf_factory = btree_index._gcchk_factory
        setattr(self, index_type + '_index', index)

    def __lt__(self, other):
        if not isinstance(other, Pack):
            raise TypeError(other)
        return id(self) < id(other)

    def __hash__(self):
        return hash((type(self), self.revision_index, self.inventory_index, self.text_index, self.signature_index, self.chk_index))