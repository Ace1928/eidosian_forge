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
def _write_index(self, index_type, index, label, suspend=False):
    """Write out an index.

        :param index_type: The type of index to write - e.g. 'revision'.
        :param index: The index object to serialise.
        :param label: What label to give the index e.g. 'revision'.
        """
    index_name = self.index_name(index_type, self.name)
    if suspend:
        transport = self.upload_transport
    else:
        transport = self.index_transport
    index_tempfile = index.finish()
    index_bytes = index_tempfile.read()
    write_stream = transport.open_write_stream(index_name, mode=self._file_mode)
    write_stream.write(index_bytes)
    write_stream.close(want_fdatasync=self._pack_collection.config_stack.get('repository.fdatasync'))
    self.index_sizes[self.index_offset(index_type)] = len(index_bytes)
    if 'pack' in debug.debug_flags:
        mutter('%s: create_pack: wrote %s index: %s%s t+%6.3fs', time.ctime(), label, self.upload_transport.base, self.random_name, time.time() - self.start_time)
    self._replace_index_with_readonly(index_type)