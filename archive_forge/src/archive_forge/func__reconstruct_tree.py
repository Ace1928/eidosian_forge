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
def _reconstruct_tree(self, fileid, revid, bzr_tree, unusual_modes, expected_sha=None):
    """Return a Git Tree object from a file id and a revision stored in bzr.

        :param fileid: fileid in the tree.
        :param revision: Revision of the tree.
        """

    def get_ie_sha1(path, entry):
        if entry.kind == 'directory':
            try:
                return self._cache.idmap.lookup_tree_id(entry.file_id, revid)
            except (NotImplementedError, KeyError):
                obj = self._reconstruct_tree(entry.file_id, revid, bzr_tree, unusual_modes)
                if obj is None:
                    return None
                else:
                    return obj.id
        elif entry.kind in ('file', 'symlink'):
            try:
                return self._cache.idmap.lookup_blob_id(entry.file_id, entry.revision)
            except KeyError:
                return next(self._reconstruct_blobs([(entry.file_id, entry.revision, None)])).id
        elif entry.kind == 'tree-reference':
            return self._lookup_revision_sha1(entry.reference_revision)
        else:
            raise AssertionError("unknown entry kind '%s'" % entry.kind)
    path = bzr_tree.id2path(fileid)
    tree = directory_to_tree(path, bzr_tree.iter_child_entries(path), get_ie_sha1, unusual_modes, self.mapping.BZR_DUMMY_FILE, bzr_tree.path2id('') == fileid)
    if tree is not None:
        _check_expected_sha(expected_sha, tree)
    return tree