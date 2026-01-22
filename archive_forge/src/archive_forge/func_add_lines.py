import gzip
import os
from io import BytesIO
from ...lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from ... import debug, errors, lockable_files, lockdir, osutils, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...bzr import tuned_gzip, versionedfile, weave, weavefile
from ...bzr.repository import RepositoryFormatMetaDir
from ...bzr.versionedfile import (AbsentContentFactory, FulltextContentFactory,
from ...bzr.vf_repository import (InterSameDataRepository,
from ...repository import InterRepository
from . import bzrdir as weave_bzrdir
from .store.text import TextStore
def add_lines(self, key, parents, lines):
    """Add a revision to the store."""
    if not self._is_locked():
        raise errors.ObjectNotLocked(self)
    if not self._can_write():
        raise errors.ReadOnlyError(self)
    if b'/' in key[-1]:
        raise ValueError('bad idea to put / in {!r}'.format(key))
    chunks = lines
    if self._compressed:
        chunks = tuned_gzip.chunks_to_gzip(chunks)
    path = self._map(key)
    self._transport.put_file_non_atomic(path, BytesIO(b''.join(chunks)), create_parent_dir=True)