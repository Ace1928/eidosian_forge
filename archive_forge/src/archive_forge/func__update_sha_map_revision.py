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
def _update_sha_map_revision(self, revid):
    rev = self.repository.get_revision(revid)
    tree = self.tree_cache.revision_tree(rev.revision_id)
    updater = self._get_updater(rev)
    for path, obj in self._revision_to_objects(rev, tree, lossy=not self.mapping.roundtripping, add_cache_entry=updater.add_object):
        if isinstance(obj, Commit):
            commit_obj = obj
    commit_obj = updater.finish()
    return commit_obj.id