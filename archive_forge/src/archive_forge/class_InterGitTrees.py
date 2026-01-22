import errno
import os
import posixpath
import stat
from collections import deque
from functools import partial
from io import BytesIO
from typing import Union, List, Tuple, Set
from dulwich.config import ConfigFile as GitConfigFile
from dulwich.config import parse_submodules
from dulwich.diff_tree import RenameDetector, tree_changes
from dulwich.errors import NotTreeError
from dulwich.index import (Index, IndexEntry, blob_from_path_and_stat,
from dulwich.object_store import OverlayObjectStore, iter_tree_contents, BaseObjectStore
from dulwich.objects import S_IFGITLINK, S_ISGITLINK, ZERO_SHA, Blob, Tree, ObjectID
from .. import controldir as _mod_controldir
from .. import delta, errors, mutabletree, osutils, revisiontree, trace
from .. import transport as _mod_transport
from .. import tree as _mod_tree
from .. import urlutils, workingtree
from ..bzr.inventorytree import InventoryTreeChange
from ..revision import CURRENT_REVISION, NULL_REVISION
from ..transport import get_transport
from ..tree import MissingNestedTree, TreeEntry
from .mapping import (decode_git_path, default_mapping, encode_git_path,
class InterGitTrees(_mod_tree.InterTree):
    """InterTree that works between two git trees."""
    _test_mutable_trees_to_test_trees = None

    def __init__(self, source, target):
        super().__init__(source, target)
        if self.source.store == self.target.store:
            self.store = self.source.store
        else:
            self.store = OverlayObjectStore([self.source.store, self.target.store])
        self.rename_detector = RenameDetector(self.store)

    @classmethod
    def is_compatible(cls, source, target):
        return isinstance(source, GitTree) and isinstance(target, GitTree)

    def compare(self, want_unchanged=False, specific_files=None, extra_trees=None, require_versioned=False, include_root=False, want_unversioned=False):
        with self.lock_read():
            changes, source_extras, target_extras = self._iter_git_changes(want_unchanged=want_unchanged, require_versioned=require_versioned, specific_files=specific_files, extra_trees=extra_trees, want_unversioned=want_unversioned)
            return tree_delta_from_git_changes(changes, (self.source.mapping, self.target.mapping), specific_files=specific_files, include_root=include_root, source_extras=source_extras, target_extras=target_extras)

    def iter_changes(self, include_unchanged=False, specific_files=None, pb=None, extra_trees=[], require_versioned=True, want_unversioned=False):
        with self.lock_read():
            changes, source_extras, target_extras = self._iter_git_changes(want_unchanged=include_unchanged, require_versioned=require_versioned, specific_files=specific_files, extra_trees=extra_trees, want_unversioned=want_unversioned)
            return changes_from_git_changes(changes, self.target.mapping, specific_files=specific_files, include_unchanged=include_unchanged, source_extras=source_extras, target_extras=target_extras)

    def _iter_git_changes(self, want_unchanged=False, specific_files=None, require_versioned=False, extra_trees=None, want_unversioned=False, include_trees=True):
        trees = [self.source]
        if extra_trees is not None:
            trees.extend(extra_trees)
        if specific_files is not None:
            specific_files = self.target.find_related_paths_across_trees(specific_files, trees, require_versioned=require_versioned)
        with self.lock_read():
            from_tree_sha, from_extras = self.source.git_snapshot(want_unversioned=want_unversioned)
            to_tree_sha, to_extras = self.target.git_snapshot(want_unversioned=want_unversioned)
            changes = tree_changes(self.store, from_tree_sha, to_tree_sha, include_trees=include_trees, rename_detector=self.rename_detector, want_unchanged=want_unchanged, change_type_same=True)
            return (changes, from_extras, to_extras)

    def find_target_path(self, path, recurse='none'):
        ret = self.find_target_paths([path], recurse=recurse)
        return ret[path]

    def find_source_path(self, path, recurse='none'):
        ret = self.find_source_paths([path], recurse=recurse)
        return ret[path]

    def find_target_paths(self, paths, recurse='none'):
        paths = set(paths)
        ret = {}
        changes = self._iter_git_changes(specific_files=paths, include_trees=False)[0]
        for change_type, old, new in changes:
            if old[0] is None:
                continue
            oldpath = decode_git_path(old[0])
            if oldpath in paths:
                ret[oldpath] = decode_git_path(new[0]) if new[0] else None
        for path in paths:
            if path not in ret:
                if self.source.has_filename(path):
                    if self.target.has_filename(path):
                        ret[path] = path
                    else:
                        ret[path] = None
                else:
                    raise _mod_transport.NoSuchFile(path)
        return ret

    def find_source_paths(self, paths, recurse='none'):
        paths = set(paths)
        ret = {}
        changes = self._iter_git_changes(specific_files=paths, include_trees=False)[0]
        for change_type, old, new in changes:
            if new[0] is None:
                continue
            newpath = decode_git_path(new[0])
            if newpath in paths:
                ret[newpath] = decode_git_path(old[0]) if old[0] else None
        for path in paths:
            if path not in ret:
                if self.target.has_filename(path):
                    if self.source.has_filename(path):
                        ret[path] = path
                    else:
                        ret[path] = None
                else:
                    raise _mod_transport.NoSuchFile(path)
        return ret