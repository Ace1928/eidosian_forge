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
class RevisionTextStore(TextVersionedFiles):
    """Legacy thunk for format 4 repositories."""

    def __init__(self, transport, serializer, compressed, mapper, is_locked, can_write):
        """Create a RevisionTextStore at transport with serializer."""
        TextVersionedFiles.__init__(self, transport, compressed, mapper, is_locked, can_write)
        self._serializer = serializer

    def _load_text_parents(self, key):
        text = self._load_text(key)
        if text is None:
            return (None, None)
        parents = self._serializer.read_revision_from_string(text).parent_ids
        return (text, tuple(((parent,) for parent in parents)))

    def get_parent_map(self, keys):
        result = {}
        for key in keys:
            parents = self._load_text_parents(key)[1]
            if parents is None:
                continue
            result[key] = parents
        return result

    def get_known_graph_ancestry(self, keys):
        """Get a KnownGraph instance with the ancestry of keys."""
        keys = self.keys()
        parent_map = self.get_parent_map(keys)
        kg = _mod_graph.KnownGraph(parent_map)
        return kg

    def get_record_stream(self, keys, sort_order, include_delta_closure):
        for key in keys:
            text, parents = self._load_text_parents(key)
            if text is None:
                yield AbsentContentFactory(key)
            else:
                yield FulltextContentFactory(key, parents, None, text)

    def keys(self):
        if not self._is_locked():
            raise errors.ObjectNotLocked(self)
        relpaths = set()
        for quoted_relpath in self._transport.iter_files_recursive():
            relpath = urlutils.unquote(quoted_relpath)
            path, ext = os.path.splitext(relpath)
            if ext == '.gz':
                relpath = path
            if not relpath.endswith('.sig'):
                relpaths.add(relpath)
        paths = list(relpaths)
        return {self._mapper.unmap(path) for path in paths}