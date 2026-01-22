import errno
import os
import re
from ..lazy_import import lazy_import
from breezy import (
from .. import transport as _mod_transport
from ..conflicts import Conflict as BaseConflict
from ..conflicts import ConflictList as BaseConflictList
from . import rio
class PathConflict(Conflict):
    """A conflict was encountered merging file paths"""
    typestring = 'path conflict'
    format = 'Path conflict: %(path)s / %(conflict_path)s'
    rformat = '%(class)s(%(path)r, %(conflict_path)r, %(file_id)r)'

    def __init__(self, path, conflict_path=None, file_id=None):
        Conflict.__init__(self, path, file_id)
        self.conflict_path = conflict_path

    def as_stanza(self):
        s = Conflict.as_stanza(self)
        if self.conflict_path is not None:
            s.add('conflict_path', self.conflict_path)
        return s

    def associated_filenames(self):
        return []

    def _resolve(self, tt, file_id, path, winner):
        """Resolve the conflict.

        :param tt: The TreeTransform where the conflict is resolved.
        :param file_id: The retained file id.
        :param path: The retained path.
        :param winner: 'this' or 'other' indicates which side is the winner.
        """
        path_to_create = None
        if winner == 'this':
            if self.path == '<deleted>':
                return
            if self.conflict_path == '<deleted>':
                path_to_create = self.path
                revid = tt._tree.get_parent_ids()[0]
        elif winner == 'other':
            if self.conflict_path == '<deleted>':
                return
            if self.path == '<deleted>':
                path_to_create = self.conflict_path
                revid = tt._tree.get_parent_ids()[-1]
        else:
            raise AssertionError('bad winner: {!r}'.format(winner))
        if path_to_create is not None:
            tid = tt.trans_id_tree_path(path_to_create)
            tree = self._revision_tree(tt._tree, revid)
            transform.create_from_tree(tt, tid, tree, tree.id2path(file_id))
            tt.version_file(tid, file_id=file_id)
        else:
            tid = tt.trans_id_file_id(file_id)
        parent_tid = tt.get_tree_parent(tid)
        tt.adjust_path(osutils.basename(path), parent_tid, tid)
        tt.apply()

    def _revision_tree(self, tree, revid):
        return tree.branch.repository.revision_tree(revid)

    def _infer_file_id(self, tree):
        possible_paths = []
        for p in (self.path, self.conflict_path):
            if p == '<deleted>':
                continue
            if p is not None:
                possible_paths.append(p)
        file_id = None
        for revid in tree.get_parent_ids():
            revtree = self._revision_tree(tree, revid)
            for p in possible_paths:
                file_id = revtree.path2id(p)
                if file_id is not None:
                    return (revtree, file_id)
        return (None, None)

    def action_take_this(self, tree):
        if self.file_id is not None:
            self._resolve_with_cleanups(tree, self.file_id, self.path, winner='this')
        else:
            revtree, file_id = self._infer_file_id(tree)
            tree.revert([revtree.id2path(file_id)], old_tree=revtree, backups=False)

    def action_take_other(self, tree):
        if self.file_id is not None:
            self._resolve_with_cleanups(tree, self.file_id, self.conflict_path, winner='other')
        else:
            revtree, file_id = self._infer_file_id(tree)
            tree.revert([revtree.id2path(file_id)], old_tree=revtree, backups=False)