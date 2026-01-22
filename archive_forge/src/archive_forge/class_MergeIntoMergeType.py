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
class MergeIntoMergeType(Merge3Merger):
    """Merger that incorporates a tree (or part of a tree) into another."""

    def __init__(self, *args, **kwargs):
        """Initialize the merger object.

        :param args: See Merge3Merger.__init__'s args.
        :param kwargs: See Merge3Merger.__init__'s keyword args, except for
            source_subpath and target_subdir.
        :keyword source_subpath: The relative path specifying the subtree of
            other_tree to merge into this_tree.
        :keyword target_subdir: The relative path where we want to merge
            other_tree into this_tree
        """
        self._source_subpath = kwargs.pop('source_subpath')
        self._target_subdir = kwargs.pop('target_subdir')
        super().__init__(*args, **kwargs)

    def _compute_transform(self):
        with ui.ui_factory.nested_progress_bar() as child_pb:
            entries = self._entries_to_incorporate()
            entries = list(entries)
            for num, (entry, parent_id, relpath) in enumerate(entries):
                child_pb.update(gettext('Preparing file merge'), num, len(entries))
                parent_trans_id = self.tt.trans_id_file_id(parent_id)
                path = osutils.pathjoin(self._source_subpath, relpath)
                trans_id = transform.new_by_entry(path, self.tt, entry, parent_trans_id, self.other_tree)
        self._finish_computing_transform()

    def _entries_to_incorporate(self):
        """Yields pairs of (inventory_entry, new_parent)."""
        subdir_id = self.other_tree.path2id(self._source_subpath)
        if subdir_id is None:
            raise PathNotInTree(self._source_subpath, 'Source tree')
        subdir = next(self.other_tree.iter_entries_by_dir(specific_files=[self._source_subpath]))[1]
        parent_in_target = osutils.dirname(self._target_subdir)
        target_id = self.this_tree.path2id(parent_in_target)
        if target_id is None:
            raise PathNotInTree(self._target_subdir, 'Target tree')
        name_in_target = osutils.basename(self._target_subdir)
        merge_into_root = subdir.copy()
        merge_into_root.name = name_in_target
        try:
            self.this_tree.id2path(merge_into_root.file_id)
        except errors.NoSuchId:
            pass
        else:
            merge_into_root.file_id = generate_ids.gen_file_id(name_in_target)
        yield (merge_into_root, target_id, '')
        if subdir.kind != 'directory':
            return
        for path, entry in self.other_tree.root_inventory.iter_entries_by_dir(subdir_id):
            parent_id = entry.parent_id
            if parent_id == subdir.file_id:
                parent_id = merge_into_root.file_id
            yield (entry, parent_id, path)