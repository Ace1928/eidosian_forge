import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
def _do_merge_contents(self, paths, trans_id, file_id):
    """Performs a merge on file_id contents."""

    def contents_pair(tree, path):
        if path is None:
            return (None, None)
        try:
            kind = tree.kind(path)
        except _mod_transport.NoSuchFile:
            return (None, None)
        if kind == 'file':
            contents = tree.get_file_sha1(path)
        elif kind == 'symlink':
            contents = tree.get_symlink_target(path)
        else:
            contents = None
        return (kind, contents)
    base_path, other_path, this_path = paths
    other_pair = contents_pair(self.other_tree, other_path)
    this_pair = contents_pair(self.this_tree, this_path)
    if self._lca_trees:
        base_path, lca_paths = base_path
        base_pair = contents_pair(self.base_tree, base_path)
        lca_pairs = [contents_pair(tree, path) for tree, path in zip(self._lca_trees, lca_paths)]
        winner = self._lca_multi_way((base_pair, lca_pairs), other_pair, this_pair, allow_overriding_lca=False)
    else:
        base_pair = contents_pair(self.base_tree, base_path)
        if base_pair == other_pair:
            winner = 'this'
        else:
            this_pair = contents_pair(self.this_tree, this_path)
            winner = self._three_way(base_pair, other_pair, this_pair)
    if winner == 'this':
        return 'unmodified'
    params = MergeFileHookParams(self, (base_path, other_path, this_path), trans_id, this_pair[0], other_pair[0], winner)
    hooks = self.active_hooks
    hook_status = 'not_applicable'
    for hook in hooks:
        hook_status, lines = hook.merge_contents(params)
        if hook_status != 'not_applicable':
            break
    keep_this = False
    result = 'modified'
    if hook_status == 'not_applicable':
        result = None
        name = self.tt.final_name(trans_id)
        parent_id = self.tt.final_parent(trans_id)
        inhibit_content_conflict = False
        if params.this_kind is None:
            if self.this_tree.is_versioned(other_path):
                keep_this = True
                self.tt.version_file(trans_id, file_id=file_id)
                transform.create_from_tree(self.tt, trans_id, self.other_tree, other_path, filter_tree_path=self._get_filter_tree_path(other_path))
                inhibit_content_conflict = True
        elif params.other_kind is None:
            if self.other_tree.is_versioned(this_path):
                keep_this = True
                inhibit_content_conflict = True
        if not inhibit_content_conflict:
            if params.this_kind is not None:
                self.tt.unversion_file(trans_id)
            file_group = self._dump_conflicts(name, (base_path, other_path, this_path), parent_id)
            for tid in file_group:
                self.tt.version_file(tid, file_id=file_id)
                break
            self._raw_conflicts.append(('contents conflict', file_group))
    elif hook_status == 'success':
        self.tt.create_file(lines, trans_id)
    elif hook_status == 'conflicted':
        self.tt.create_file(lines, trans_id)
        self._raw_conflicts.append(('text conflict', trans_id))
        name = self.tt.final_name(trans_id)
        parent_id = self.tt.final_parent(trans_id)
        self._dump_conflicts(name, (base_path, other_path, this_path), parent_id)
    elif hook_status == 'delete':
        self.tt.unversion_file(trans_id)
        result = 'deleted'
    elif hook_status == 'done':
        pass
    else:
        raise AssertionError('unknown hook_status: {!r}'.format(hook_status))
    if not this_path and result == 'modified':
        self.tt.version_file(trans_id, file_id=file_id)
    if not keep_this:
        self.tt.delete_contents(trans_id)
    return result