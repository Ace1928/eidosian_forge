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
def _get_filecommands(self, tree_old, tree_new):
    """Get the list of FileCommands for the changes between two revisions."""
    changes = tree_new.changes_from(tree_old)
    my_modified = list(changes.modified)
    file_cmds, rd_modifies, renamed = self._process_renames_and_deletes(changes.renamed, changes.removed, tree_new.get_revision_id(), tree_old)
    yield from file_cmds
    for change in changes.kind_changed:
        path = self._adjust_path_for_renames(change.path[0], renamed, tree_new.get_revision_id())
        my_modified.append(change)
    files_to_get = []
    for change in changes.added + changes.copied + my_modified + rd_modifies:
        if change.kind[1] == 'file':
            files_to_get.append((change.path[1], (change.path[1], helpers.kind_to_mode('file', change.executable[1]))))
        elif change.kind[1] == 'symlink':
            yield commands.FileModifyCommand(change.path[1].encode('utf-8'), helpers.kind_to_mode('symlink', False), None, tree_new.get_symlink_target(change.path[1]).encode('utf-8'))
        elif change.kind[1] == 'directory':
            if not self.plain_format:
                yield commands.FileModifyCommand(change.path[1].encode('utf-8'), helpers.kind_to_mode('directory', False), None, None)
        else:
            self.warning("cannot export '%s' of kind %s yet - ignoring" % (change.path[1], change.kind[1]))
    for (path, mode), chunks in tree_new.iter_files_bytes(files_to_get):
        yield commands.FileModifyCommand(path.encode('utf-8'), mode, None, b''.join(chunks))