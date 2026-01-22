import posixpath
import stat
from typing import Dict, Iterable, Iterator, List
from dulwich.object_store import BaseObjectStore
from dulwich.objects import (ZERO_SHA, Blob, Commit, ObjectID, ShaFile, Tree,
from dulwich.pack import Pack, PackData, pack_objects_to_data
from .. import errors, lru_cache, osutils, trace, ui
from ..bzr.testament import StrictTestament3
from ..lock import LogicalLockResult
from ..revision import NULL_REVISION
from ..tree import InterTree
from .cache import from_repository as cache_from_repository
from .mapping import (default_mapping, encode_git_path, entry_mode,
from .unpeel_map import UnpeelMap
def _lookup_revision_sha1(self, revid):
    """Return the SHA1 matching a Bazaar revision."""
    if revid == NULL_REVISION:
        return ZERO_SHA
    try:
        return self._cache.idmap.lookup_commit(revid)
    except KeyError:
        try:
            return mapping_registry.parse_revision_id(revid)[0]
        except errors.InvalidRevisionId:
            self._update_sha_map(revid)
            return self._cache.idmap.lookup_commit(revid)