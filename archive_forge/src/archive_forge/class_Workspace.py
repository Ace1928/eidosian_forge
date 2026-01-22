import errno
import os
import shutil
from contextlib import ExitStack
from typing import List, Optional
from .clean_tree import iter_deletables
from .errors import BzrError, DependencyNotPresent
from .osutils import is_inside
from .trace import warning
from .transform import revert
from .transport import NoSuchFile
from .tree import Tree
from .workingtree import WorkingTree
class Workspace:
    """Create a workspace.

    :param tree: Tree to work in
    :param subpath: path under which to consider and commit changes
    :param use_inotify: whether to use inotify (default: yes, if available)
    """

    def __init__(self, tree, subpath='', use_inotify=None):
        self.tree = tree
        self.subpath = subpath
        self.use_inotify = use_inotify
        self._dirty_tracker = None
        self._es = ExitStack()

    @classmethod
    def from_path(cls, path, use_inotify=None):
        tree, subpath = WorkingTree.open_containing(path)
        return cls(tree, subpath, use_inotify=use_inotify)

    def __enter__(self):
        check_clean_tree(self.tree)
        self._es.__enter__()
        self._dirty_tracker = get_dirty_tracker(self.tree, subpath=self.subpath, use_inotify=self.use_inotify)
        if self._dirty_tracker:
            from .dirty_tracker import TooManyOpenFiles
            try:
                self._es.enter_context(self._dirty_tracker)
            except TooManyOpenFiles:
                warning('Too many files open; not using inotify')
                self._dirty_tracker = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._es.__exit__(exc_type, exc_val, exc_tb)

    def tree_path(self, path=''):
        """Return a path relative to the tree subpath used by this workspace.
        """
        return os.path.join(self.subpath, path)

    def abspath(self, path=''):
        """Return an absolute path for the tree."""
        return self.tree.abspath(self.tree_path(path))

    def reset(self):
        """Reset - revert local changes, revive deleted files, remove added.
        """
        if self._dirty_tracker and (not self._dirty_tracker.is_dirty()):
            return
        reset_tree(self.tree, subpath=self.subpath)
        if self._dirty_tracker is not None:
            self._dirty_tracker.mark_clean()

    def _stage(self) -> Optional[List[str]]:
        changed: Optional[List[str]]
        if self._dirty_tracker:
            relpaths = self._dirty_tracker.relpaths()
            self.tree.add([p for p in sorted(relpaths) if self.tree.has_filename(p) and (not self.tree.is_ignored(p))])
            changed = [p for p in relpaths if self.tree.is_versioned(p)]
        else:
            self.tree.smart_add([self.tree.abspath(self.subpath)])
            changed = [self.subpath] if self.subpath else None
        if self.tree.supports_setting_file_ids():
            from .rename_map import RenameMap
            basis_tree = self.tree.basis_tree()
            RenameMap.guess_renames(basis_tree, self.tree, dry_run=False)
        return changed

    def iter_changes(self):
        with self.tree.lock_write():
            specific_files = self._stage()
            basis_tree = self.tree.basis_tree()
            for change in self.tree.iter_changes(basis_tree, specific_files=specific_files, want_unversioned=False, require_versioned=True):
                if change.kind[1] is None and change.versioned[1]:
                    if change.path[0] is None:
                        continue
                    change = change.discard_new()
                yield change

    def commit(self, **kwargs):
        """Create a commit.

        See WorkingTree.commit() for documentation.
        """
        if 'specific_files' in kwargs:
            raise NotImplementedError(self.commit)
        with self.tree.lock_write():
            specific_files = self._stage()
            kwargs['specific_files'] = specific_files
            revid = self.tree.commit(**kwargs)
            if self._dirty_tracker:
                self._dirty_tracker.mark_clean()
            return revid