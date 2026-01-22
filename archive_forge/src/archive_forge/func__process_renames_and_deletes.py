import re
import sys
import time
from email.utils import parseaddr
import breezy.branch
import breezy.revision
from ... import (builtins, errors, lazy_import, lru_cache, osutils, progress,
from ... import transport as _mod_transport
from . import helpers, marks_file
from fastimport import commands
def _process_renames_and_deletes(self, renames, deletes, revision_id, tree_old):
    file_cmds = []
    modifies = []
    renamed = []
    must_be_renamed = {}
    old_to_new = {}
    deleted_paths = {change.path[0] for change in deletes}
    for change in renames:
        emit = change.kind[1] != 'directory' or not self.plain_format
        if change.path[1] in deleted_paths:
            if emit:
                file_cmds.append(commands.FileDeleteCommand(change.path[1].encode('utf-8')))
            deleted_paths.remove(change.path[1])
        if self.is_empty_dir(tree_old, change.path[0]):
            self.note('Skipping empty dir {} in rev {}'.format(change.path[0], revision_id))
            continue
        renamed.append(change.path)
        old_to_new[change.path[0]] = change.path[1]
        if emit:
            file_cmds.append(commands.FileRenameCommand(change.path[0].encode('utf-8'), change.path[1].encode('utf-8')))
        if change.changed_content or change.meta_modified():
            modifies.append(change)
        if change.kind == ('directory', 'directory'):
            for p, e in tree_old.iter_entries_by_dir(specific_files=[change.path[0]]):
                if e.kind == 'directory' and self.plain_format:
                    continue
                old_child_path = osutils.pathjoin(change.path[0], p)
                new_child_path = osutils.pathjoin(change.path[1], p)
                must_be_renamed[old_child_path] = new_child_path
    if must_be_renamed:
        renamed_already = set(old_to_new.keys())
        still_to_be_renamed = set(must_be_renamed.keys()) - renamed_already
        for old_child_path in sorted(still_to_be_renamed):
            new_child_path = must_be_renamed[old_child_path]
            if self.verbose:
                self.note('implicitly renaming {} => {}'.format(old_child_path, new_child_path))
            file_cmds.append(commands.FileRenameCommand(old_child_path.encode('utf-8'), new_child_path.encode('utf-8')))
    for change in deletes:
        if change.path[0] not in deleted_paths:
            continue
        if change.kind[0] == 'directory' and self.plain_format:
            continue
        file_cmds.append(commands.FileDeleteCommand(change.path[0].encode('utf-8')))
    return (file_cmds, modifies, renamed)