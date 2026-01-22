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
class ResumedPack(ExistingPack):

    def __init__(self, name, revision_index, inventory_index, text_index, signature_index, upload_transport, pack_transport, index_transport, pack_collection, chk_index=None):
        """Create a ResumedPack object."""
        ExistingPack.__init__(self, pack_transport, name, revision_index, inventory_index, text_index, signature_index, chk_index=chk_index)
        self.upload_transport = upload_transport
        self.index_transport = index_transport
        self.index_sizes = [None, None, None, None]
        indices = [('revision', revision_index), ('inventory', inventory_index), ('text', text_index), ('signature', signature_index)]
        if chk_index is not None:
            indices.append(('chk', chk_index))
            self.index_sizes.append(None)
        for index_type, index in indices:
            offset = self.index_offset(index_type)
            self.index_sizes[offset] = index._size
        self.index_class = pack_collection._index_class
        self._pack_collection = pack_collection
        self._state = 'resumed'

    def access_tuple(self):
        if self._state == 'finished':
            return Pack.access_tuple(self)
        elif self._state == 'resumed':
            return (self.upload_transport, self.file_name())
        else:
            raise AssertionError(self._state)

    def abort(self):
        self.upload_transport.delete(self.file_name())
        indices = [self.revision_index, self.inventory_index, self.text_index, self.signature_index]
        if self.chk_index is not None:
            indices.append(self.chk_index)
        for index in indices:
            index._transport.delete(index._name)

    def finish(self):
        self._check_references()
        index_types = ['revision', 'inventory', 'text', 'signature']
        if self.chk_index is not None:
            index_types.append('chk')
        for index_type in index_types:
            old_name = self.index_name(index_type, self.name)
            new_name = '../indices/' + old_name
            self.upload_transport.move(old_name, new_name)
            self._replace_index_with_readonly(index_type)
        new_name = '../packs/' + self.file_name()
        self.upload_transport.move(self.file_name(), new_name)
        self._state = 'finished'

    def _get_external_refs(self, index):
        """Return compression parents for this index that are not present.

        This returns any compression parents that are referenced by this index,
        which are not contained *in* this index. They may be present elsewhere.
        """
        return index.external_references(1)