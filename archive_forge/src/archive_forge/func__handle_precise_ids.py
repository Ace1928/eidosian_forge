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