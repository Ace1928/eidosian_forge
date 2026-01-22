import os
import re
from collections import deque
from typing import TYPE_CHECKING, Optional, Type
from .. import branch as _mod_branch
from .. import controldir, debug, errors, lazy_import, osutils, revision, trace
from .. import transport as _mod_transport
from ..controldir import ControlDir
from ..mutabletree import MutableTree
from ..repository import Repository
from ..revisiontree import RevisionTree
from breezy import (
from breezy.bzr import (
from ..tree import (FileTimestampUnavailable, InterTree, MissingNestedTree,
class InventoryRevisionTree(RevisionTree, InventoryTree):

    def __init__(self, repository, inv, revision_id):
        RevisionTree.__init__(self, repository, revision_id)
        self._inventory = inv

    def _get_file_revision(self, path, file_id, vf, tree_revision):
        """Ensure that file_id, tree_revision is in vf to plan the merge."""
        last_revision = self.get_file_revision(path)
        base_vf = self._repository.texts
        if base_vf not in vf.fallback_versionedfiles:
            vf.fallback_versionedfiles.append(base_vf)
        return last_revision

    def get_file_mtime(self, path):
        ie = self._path2ie(path)
        try:
            revision = self._repository.get_revision(ie.revision)
        except errors.NoSuchRevision:
            raise FileTimestampUnavailable(path)
        return revision.timestamp

    def get_file_size(self, path):
        return self._path2ie(path).text_size

    def get_file_sha1(self, path, stat_value=None):
        ie = self._path2ie(path)
        if ie.kind == 'file':
            return ie.text_sha1
        return None

    def get_file_revision(self, path):
        return self._path2ie(path).revision

    def is_executable(self, path):
        ie = self._path2ie(path)
        if ie.kind != 'file':
            return False
        return ie.executable

    def has_filename(self, filename):
        return bool(self.path2id(filename))

    def reference_parent(self, path, branch=None, possible_transports=None):
        if branch is not None:
            file_id = self.path2id(path)
            parent_url = branch.get_reference_info(file_id)[0]
        else:
            subdir = ControlDir.open_from_transport(self._repository.user_transport.clone(path))
            parent_url = subdir.open_branch().get_parent()
        if parent_url is None:
            return None
        return _mod_branch.Branch.open(parent_url, possible_transports=possible_transports)

    def get_reference_info(self, path, branch=None):
        return branch.get_reference_info(self.path2id(path))[0]

    def list_files(self, include_root=False, from_dir=None, recursive=True, recurse_nested=False):
        if from_dir is None:
            from_dir_id = None
            inv = self.root_inventory
        else:
            inv, from_dir_id = self._path2inv_file_id(from_dir)
            if from_dir_id is None:
                return
        entries = inv.iter_entries(from_dir=from_dir_id, recursive=recursive)
        if inv.root is not None and (not include_root) and (from_dir is None):
            next(entries)
        for path, entry in entries:
            if entry.kind == 'tree-reference' and recurse_nested:
                subtree = self._get_nested_tree(path, entry.file_id, entry.reference_revision)
                for subpath, status, kind, entry in subtree.list_files(include_root=True, recurse_nested=recurse_nested, recursive=recursive):
                    if subpath:
                        full_subpath = osutils.pathjoin(path, subpath)
                    else:
                        full_subpath = path
                    yield (full_subpath, status, kind, entry)
            else:
                yield (path, 'V', entry.kind, entry)

    def get_symlink_target(self, path):
        return self._path2ie(path).symlink_target

    def get_reference_revision(self, path):
        return self._path2ie(path).reference_revision

    def _get_nested_tree(self, path, file_id, reference_revision):
        try:
            subdir = ControlDir.open_from_transport(self._repository.user_transport.clone(path))
        except errors.NotBranchError as e:
            raise MissingNestedTree(path) from e
        subrepo = subdir.find_repository()
        try:
            revtree = subrepo.revision_tree(reference_revision)
        except errors.NoSuchRevision:
            raise MissingNestedTree(path)
        if file_id is not None and file_id != revtree.path2id(''):
            raise AssertionError('invalid root id: {!r} != {!r}'.format(file_id, revtree.path2id('')))
        return revtree

    def get_nested_tree(self, path):
        nested_revid = self.get_reference_revision(path)
        return self._get_nested_tree(path, None, nested_revid)

    def kind(self, path):
        return self._path2ie(path).kind

    def path_content_summary(self, path):
        """See Tree.path_content_summary."""
        try:
            entry = self._path2ie(path)
        except _mod_transport.NoSuchFile:
            return ('missing', None, None, None)
        kind = entry.kind
        if kind == 'file':
            return (kind, entry.text_size, entry.executable, entry.text_sha1)
        elif kind == 'symlink':
            return (kind, None, None, entry.symlink_target)
        else:
            return (kind, None, None, None)

    def _comparison_data(self, entry, path):
        if entry is None:
            return (None, False, None)
        return (entry.kind, entry.executable, None)

    def walkdirs(self, prefix=''):
        _directory = 'directory'
        inv, top_id = self._path2inv_file_id(prefix)
        if top_id is None:
            pending = []
        else:
            pending = [(prefix, top_id)]
        while pending:
            dirblock = []
            root, file_id = pending.pop()
            if root:
                relroot = root + '/'
            else:
                relroot = ''
            entry = inv.get_entry(file_id)
            subdirs = []
            for name, child in entry.sorted_children():
                toppath = relroot + name
                dirblock.append((toppath, name, child.kind, None, child.kind))
                if child.kind == _directory:
                    subdirs.append((toppath, child.file_id))
            yield (root, dirblock)
            pending.extend(reversed(subdirs))

    def iter_files_bytes(self, desired_files):
        """See Tree.iter_files_bytes.

        This version is implemented on top of Repository.iter_files_bytes"""
        repo_desired_files = [(self.path2id(f), self.get_file_revision(f), i) for f, i in desired_files]
        try:
            yield from self._repository.iter_files_bytes(repo_desired_files)
        except errors.RevisionNotPresent as e:
            raise _mod_transport.NoSuchFile(e.file_id)

    def annotate_iter(self, path, default_revision=revision.CURRENT_REVISION):
        """See Tree.annotate_iter"""
        file_id = self.path2id(path)
        text_key = (file_id, self.get_file_revision(path))
        annotator = self._repository.texts.get_annotator()
        annotations = annotator.annotate_flat(text_key)
        return [(key[-1], line) for key, line in annotations]

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, InventoryRevisionTree):
            return self.root_inventory == other.root_inventory
        return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        raise ValueError('not hashable')