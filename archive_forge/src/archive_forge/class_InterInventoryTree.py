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
class InterInventoryTree(InterTree):
    """InterTree implementation for InventoryTree objects.

    """

    @classmethod
    def is_compatible(kls, source, target):
        return isinstance(source, InventoryTree) and isinstance(target, InventoryTree)

    def _changes_from_entries(self, source_entry, target_entry, source_path, target_path):
        """Generate a iter_changes tuple between source_entry and target_entry.

        :param source_entry: An inventory entry from self.source, or None.
        :param target_entry: An inventory entry from self.target, or None.
        :param source_path: The path of source_entry.
        :param target_path: The path of target_entry.
        :return: A tuple, item 0 of which is an iter_changes result tuple, and
            item 1 is True if there are any changes in the result tuple.
        """
        if source_entry is None:
            if target_entry is None:
                return None
            file_id = target_entry.file_id
        else:
            file_id = source_entry.file_id
        if source_entry is not None:
            source_versioned = True
            source_name = source_entry.name
            source_parent = source_entry.parent_id
            source_kind, source_executable, source_stat = self.source._comparison_data(source_entry, source_path)
        else:
            source_versioned = False
            source_name = None
            source_parent = None
            source_kind = None
            source_executable = None
        if target_entry is not None:
            target_versioned = True
            target_name = target_entry.name
            target_parent = target_entry.parent_id
            target_kind, target_executable, target_stat = self.target._comparison_data(target_entry, target_path)
        else:
            target_versioned = False
            target_name = None
            target_parent = None
            target_kind = None
            target_executable = None
        versioned = (source_versioned, target_versioned)
        kind = (source_kind, target_kind)
        changed_content = False
        if source_kind != target_kind:
            changed_content = True
        elif source_kind == 'file':
            if not self.file_content_matches(source_path, target_path, source_stat, target_stat):
                changed_content = True
        elif source_kind == 'symlink':
            if self.source.get_symlink_target(source_path) != self.target.get_symlink_target(target_path):
                changed_content = True
        elif source_kind == 'tree-reference':
            if self.source.get_reference_revision(source_path) != self.target.get_reference_revision(target_path):
                changed_content = True
        parent = (source_parent, target_parent)
        name = (source_name, target_name)
        executable = (source_executable, target_executable)
        if changed_content is not False or versioned[0] != versioned[1] or parent[0] != parent[1] or (name[0] != name[1]) or (executable[0] != executable[1]):
            changes = True
        else:
            changes = False
        return (InventoryTreeChange(file_id, (source_path, target_path), changed_content, versioned, parent, name, kind, executable), changes)

    def iter_changes(self, include_unchanged=False, specific_files=None, pb=None, extra_trees=[], require_versioned=True, want_unversioned=False):
        """Generate an iterator of changes between trees.

        A tuple is returned:
        (file_id, (path_in_source, path_in_target),
         changed_content, versioned, parent, name, kind,
         executable)

        Changed_content is True if the file's content has changed.  This
        includes changes to its kind, and to a symlink's target.

        versioned, parent, name, kind, executable are tuples of (from, to).
        If a file is missing in a tree, its kind is None.

        Iteration is done in parent-to-child order, relative to the target
        tree.

        There is no guarantee that all paths are in sorted order: the
        requirement to expand the search due to renames may result in children
        that should be found early being found late in the search, after
        lexically later results have been returned.
        :param require_versioned: Raise errors.PathsNotVersionedError if a
            path in the specific_files list is not versioned in one of
            source, target or extra_trees.
        :param specific_files: An optional list of file paths to restrict the
            comparison to. When mapping filenames to ids, all matches in all
            trees (including optional extra_trees) are used, and all children
            of matched directories are included. The parents in the target tree
            of the specific files up to and including the root of the tree are
            always evaluated for changes too.
        :param want_unversioned: Should unversioned files be returned in the
            output. An unversioned file is defined as one with (False, False)
            for the versioned pair.
        """
        if not extra_trees:
            extra_trees = []
        else:
            extra_trees = list(extra_trees)
        precise_file_ids = set()
        changed_file_ids = []
        if specific_files == []:
            target_specific_files = []
            source_specific_files = []
        else:
            target_specific_files = self.target.find_related_paths_across_trees(specific_files, [self.source] + extra_trees, require_versioned=require_versioned)
            source_specific_files = self.source.find_related_paths_across_trees(specific_files, [self.target] + extra_trees, require_versioned=require_versioned)
        if specific_files is not None:
            seen_parents = set()
            seen_dirs = set()
        if want_unversioned:
            all_unversioned = sorted([(p.split('/'), p) for p in self.target.extras() if specific_files is None or osutils.is_inside_any(specific_files, p)])
            all_unversioned = deque(all_unversioned)
        else:
            all_unversioned = deque()
        to_paths = {}
        from_entries_by_dir = list(self.source.iter_entries_by_dir(specific_files=source_specific_files))
        from_data = dict(from_entries_by_dir)
        to_entries_by_dir = list(self.target.iter_entries_by_dir(specific_files=target_specific_files))
        path_equivs = self.find_source_paths([p for p, e in to_entries_by_dir])
        num_entries = len(from_entries_by_dir) + len(to_entries_by_dir)
        entry_count = 0
        fake_entry = TreeFile()
        for target_path, target_entry in to_entries_by_dir:
            while all_unversioned and all_unversioned[0][0] < target_path.split('/'):
                unversioned_path = all_unversioned.popleft()
                target_kind, target_executable, target_stat = self.target._comparison_data(fake_entry, unversioned_path[1])
                yield InventoryTreeChange(None, (None, unversioned_path[1]), True, (False, False), (None, None), (None, unversioned_path[0][-1]), (None, target_kind), (None, target_executable))
            source_path = path_equivs[target_path]
            if source_path is not None:
                source_entry = from_data.get(source_path)
            else:
                source_entry = None
            result, changes = self._changes_from_entries(source_entry, target_entry, source_path=source_path, target_path=target_path)
            to_paths[result.file_id] = result.path[1]
            entry_count += 1
            if result.versioned[0]:
                entry_count += 1
            if pb is not None:
                pb.update('comparing files', entry_count, num_entries)
            if changes or include_unchanged:
                if specific_files is not None:
                    precise_file_ids.add(result.parent_id[1])
                    changed_file_ids.append(result.file_id)
                yield result
            if specific_files is not None:
                if result.kind[1] == 'directory':
                    seen_dirs.add(result.file_id)
                if not result.versioned[0] or result.is_reparented():
                    seen_parents.add(result.parent_id[1])
        while all_unversioned:
            unversioned_path = all_unversioned.popleft()
            to_kind, to_executable, to_stat = self.target._comparison_data(fake_entry, unversioned_path[1])
            yield InventoryTreeChange(None, (None, unversioned_path[1]), True, (False, False), (None, None), (None, unversioned_path[0][-1]), (None, to_kind), (None, to_executable))
        for path, from_entry in from_entries_by_dir:
            file_id = from_entry.file_id
            if file_id in to_paths:
                continue
            to_path = self.find_target_path(path)
            entry_count += 1
            if pb is not None:
                pb.update('comparing files', entry_count, num_entries)
            versioned = (True, False)
            parent = (from_entry.parent_id, None)
            name = (from_entry.name, None)
            from_kind, from_executable, stat_value = self.source._comparison_data(from_entry, path)
            kind = (from_kind, None)
            executable = (from_executable, None)
            changed_content = from_kind is not None
            changed_file_ids.append(file_id)
            yield InventoryTreeChange(file_id, (path, to_path), changed_content, versioned, parent, name, kind, executable)
        changed_file_ids = set(changed_file_ids)
        if specific_files is not None:
            for result in self._handle_precise_ids(precise_file_ids, changed_file_ids):
                yield result

    @staticmethod
    def _get_entry(tree, path):
        """Get an inventory entry from a tree, with missing entries as None.

        If the tree raises NotImplementedError on accessing .inventory, then
        this is worked around using iter_entries_by_dir on just the file id
        desired.

        :param tree: The tree to lookup the entry in.
        :param path: The path to look up
        """
        try:
            iterator = tree.iter_entries_by_dir(specific_files=[path])
            return next(iterator)[1]
        except StopIteration:
            return None

    def _handle_precise_ids(self, precise_file_ids, changed_file_ids, discarded_changes=None):
        """Fill out a partial iter_changes to be consistent.

        :param precise_file_ids: The file ids of parents that were seen during
            the iter_changes.
        :param changed_file_ids: The file ids of already emitted items.
        :param discarded_changes: An optional dict of precalculated
            iter_changes items which the partial iter_changes had not output
            but had calculated.
        :return: A generator of iter_changes items to output.
        """
        while precise_file_ids:
            precise_file_ids.discard(None)
            precise_file_ids.difference_update(changed_file_ids)
            if not precise_file_ids:
                break
            paths = []
            for parent_id in precise_file_ids:
                try:
                    paths.append(self.target.id2path(parent_id))
                except errors.NoSuchId:
                    pass
            for path in paths:
                old_id = self.source.path2id(path)
                precise_file_ids.add(old_id)
            precise_file_ids.discard(None)
            current_ids = precise_file_ids
            precise_file_ids = set()
            for file_id in current_ids:
                if discarded_changes:
                    result = discarded_changes.get(file_id)
                    source_entry = None
                else:
                    result = None
                if result is None:
                    try:
                        source_path = self.source.id2path(file_id)
                    except errors.NoSuchId:
                        source_path = None
                        source_entry = None
                    else:
                        source_entry = self._get_entry(self.source, source_path)
                    try:
                        target_path = self.target.id2path(file_id)
                    except errors.NoSuchId:
                        target_path = None
                        target_entry = None
                    else:
                        target_entry = self._get_entry(self.target, target_path)
                    result, changes = self._changes_from_entries(source_entry, target_entry, source_path, target_path)
                else:
                    changes = True
                new_parent_id = result.parent_id[1]
                precise_file_ids.add(new_parent_id)
                if changes:
                    if result.kind[0] == 'directory' and result.kind[1] != 'directory':
                        if source_entry is None:
                            source_entry = self._get_entry(self.source, result.path[0])
                        precise_file_ids.update((child.file_id for child in self.source.iter_child_entries(result.path[0])))
                    changed_file_ids.add(result.file_id)
                    yield result

    def find_target_path(self, path, recurse='none'):
        """Find target tree path.

        :param path: Path to search for (exists in source)
        :return: path in target, or None if there is no equivalent path.
        :raise NoSuchFile: If the path doesn't exist in source
        """
        file_id = self.source.path2id(path)
        if file_id is None:
            raise _mod_transport.NoSuchFile(path)
        try:
            return self.target.id2path(file_id, recurse=recurse)
        except errors.NoSuchId:
            return None

    def find_source_path(self, path, recurse='none'):
        """Find the source tree path.

        :param path: Path to search for (exists in target)
        :return: path in source, or None if there is no equivalent path.
        :raise NoSuchFile: if the path doesn't exist in target
        """
        file_id = self.target.path2id(path)
        if file_id is None:
            raise _mod_transport.NoSuchFile(path)
        try:
            return self.source.id2path(file_id, recurse=recurse)
        except errors.NoSuchId:
            return None