import contextlib
import errno
import os
import time
from stat import S_IEXEC, S_ISREG
from typing import Callable
from . import config as _mod_config
from . import controldir, errors, lazy_import, lock, osutils, registry, trace
from breezy import (
from breezy.i18n import gettext
from .errors import BzrError, DuplicateKey, InternalBzrError
from .filters import ContentFilterContext, filtered_output_bytes
from .mutabletree import MutableTree
from .osutils import delete_any, file_kind, pathjoin, sha_file, splitpath
from .progress import ProgressPhase
from .transport import FileExists, NoSuchFile
from .tree import InterTree, find_previous_path
class PreviewTree:
    """Preview tree."""

    def __init__(self, transform):
        self._transform = transform
        self._parent_ids = []
        self.__by_parent = None
        self._path2trans_id_cache = {}
        self._all_children_cache = {}
        self._final_name_cache = {}

    def supports_setting_file_ids(self):
        raise NotImplementedError(self.supports_setting_file_ids)

    def supports_symlinks(self):
        return self._transform._tree.supports_symlinks()

    @property
    def _by_parent(self):
        if self.__by_parent is None:
            self.__by_parent = self._transform.by_parent()
        return self.__by_parent

    def get_parent_ids(self):
        return self._parent_ids

    def set_parent_ids(self, parent_ids):
        self._parent_ids = parent_ids

    def get_revision_tree(self, revision_id):
        return self._transform._tree.get_revision_tree(revision_id)

    def is_locked(self):
        return False

    def lock_read(self):
        return lock.LogicalLockResult(self.unlock)

    def unlock(self):
        pass

    def _path2trans_id(self, path):
        """Look up the trans id associated with a path.

        :param path: path to look up, None when the path does not exist
        :return: trans_id
        """
        trans_id = self._path2trans_id_cache.get(path, object)
        if trans_id is not object:
            return trans_id
        segments = osutils.splitpath(path)
        cur_parent = self._transform.root
        for cur_segment in segments:
            for child in self._all_children(cur_parent):
                final_name = self._final_name_cache.get(child)
                if final_name is None:
                    final_name = self._transform.final_name(child)
                    self._final_name_cache[child] = final_name
                if final_name == cur_segment:
                    cur_parent = child
                    break
            else:
                self._path2trans_id_cache[path] = None
                return None
        self._path2trans_id_cache[path] = cur_parent
        return cur_parent

    def _all_children(self, trans_id):
        children = self._all_children_cache.get(trans_id)
        if children is not None:
            return children
        children = set(self._transform.iter_tree_children(trans_id))
        children.difference_update(self._transform._new_parent)
        children.update(self._by_parent.get(trans_id, []))
        self._all_children_cache[trans_id] = children
        return children

    def get_file_with_stat(self, path):
        return (self.get_file(path), None)

    def is_executable(self, path):
        trans_id = self._path2trans_id(path)
        if trans_id is None:
            return False
        try:
            return self._transform._new_executability[trans_id]
        except KeyError:
            try:
                return self._transform._tree.is_executable(path)
            except OSError as e:
                if e.errno == errno.ENOENT:
                    return False
                raise
            except NoSuchFile:
                return False

    def has_filename(self, path):
        trans_id = self._path2trans_id(path)
        if trans_id in self._transform._new_contents:
            return True
        elif trans_id in self._transform._removed_contents:
            return False
        else:
            return self._transform._tree.has_filename(path)

    def get_file_sha1(self, path, stat_value=None):
        trans_id = self._path2trans_id(path)
        if trans_id is None:
            raise NoSuchFile(path)
        kind = self._transform._new_contents.get(trans_id)
        if kind is None:
            return self._transform._tree.get_file_sha1(path)
        if kind == 'file':
            with self.get_file(path) as fileobj:
                return osutils.sha_file(fileobj)

    def get_file_verifier(self, path, stat_value=None):
        trans_id = self._path2trans_id(path)
        if trans_id is None:
            raise NoSuchFile(path)
        kind = self._transform._new_contents.get(trans_id)
        if kind is None:
            return self._transform._tree.get_file_verifier(path)
        if kind == 'file':
            with self.get_file(path) as fileobj:
                return ('SHA1', osutils.sha_file(fileobj))

    def kind(self, path):
        trans_id = self._path2trans_id(path)
        if trans_id is None:
            raise NoSuchFile(path)
        return self._transform.final_kind(trans_id)

    def stored_kind(self, path):
        trans_id = self._path2trans_id(path)
        if trans_id is None:
            raise NoSuchFile(path)
        try:
            return self._transform._new_contents[trans_id]
        except KeyError:
            return self._transform._tree.stored_kind(path)

    def _get_repository(self):
        repo = getattr(self._transform._tree, '_repository', None)
        if repo is None:
            repo = self._transform._tree.branch.repository
        return repo

    def _iter_parent_trees(self):
        for revision_id in self.get_parent_ids():
            try:
                yield self.revision_tree(revision_id)
            except errors.NoSuchRevisionInTree:
                yield self._get_repository().revision_tree(revision_id)

    def get_file_size(self, path):
        """See Tree.get_file_size"""
        trans_id = self._path2trans_id(path)
        if trans_id is None:
            raise NoSuchFile(path)
        kind = self._transform.final_kind(trans_id)
        if kind != 'file':
            return None
        if trans_id in self._transform._new_contents:
            return self._stat_limbo_file(trans_id).st_size
        if self.kind(path) == 'file':
            return self._transform._tree.get_file_size(path)
        else:
            return None

    def get_reference_revision(self, path):
        trans_id = self._path2trans_id(path)
        if trans_id is None:
            raise NoSuchFile(path)
        reference_revision = self._transform._new_reference_revision.get(trans_id)
        if reference_revision is None:
            return self._transform._tree.get_reference_revision(path)
        return reference_revision

    def tree_kind(self, trans_id):
        path = self._tree_id_paths.get(trans_id)
        if path is None:
            return None
        kind = self._tree.path_content_summary(path)[0]
        if kind == 'missing':
            kind = None
        return kind