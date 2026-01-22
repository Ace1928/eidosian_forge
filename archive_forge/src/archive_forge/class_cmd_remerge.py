import errno
import os
import sys
import breezy.bzr
import breezy.git
from . import controldir, errors, lazy_import, transport
import time
import breezy
from breezy import (
from breezy.branch import Branch
from breezy.transport import memory
from breezy.smtp_connection import SMTPConnection
from breezy.workingtree import WorkingTree
from breezy.i18n import gettext, ngettext
from .commands import Command, builtin_command_registry, display_command
from .option import (ListOption, Option, RegistryOption, _parse_revision_str,
from .revisionspec import RevisionInfo, RevisionSpec
from .trace import get_verbosity_level, is_quiet, mutter, note, warning
class cmd_remerge(Command):
    __doc__ = 'Redo a merge.\n\n    Use this if you want to try a different merge technique while resolving\n    conflicts.  Some merge techniques are better than others, and remerge\n    lets you try different ones on different files.\n\n    The options for remerge have the same meaning and defaults as the ones for\n    merge.  The difference is that remerge can (only) be run when there is a\n    pending merge, and it lets you specify particular files.\n\n    :Examples:\n        Re-do the merge of all conflicted files, and show the base text in\n        conflict regions, in addition to the usual THIS and OTHER texts::\n\n            brz remerge --show-base\n\n        Re-do the merge of "foobar", using the weave merge algorithm, with\n        additional processing to reduce the size of conflict regions::\n\n            brz remerge --merge-type weave --reprocess foobar\n    '
    takes_args = ['file*']
    takes_options = ['merge-type', 'reprocess', Option('show-base', help='Show base revision text in conflicts.')]

    def run(self, file_list=None, merge_type=None, show_base=False, reprocess=False):
        from .conflicts import restore
        if merge_type is None:
            merge_type = _mod_merge.Merge3Merger
        tree, file_list = WorkingTree.open_containing_paths(file_list)
        self.enter_context(tree.lock_write())
        parents = tree.get_parent_ids()
        if len(parents) != 2:
            raise errors.CommandError(gettext('Sorry, remerge only works after normal merges.  Not cherrypicking or multi-merges.'))
        interesting_files = None
        new_conflicts = []
        conflicts = tree.conflicts()
        if file_list is not None:
            interesting_files = set()
            for filename in file_list:
                if not tree.is_versioned(filename):
                    raise errors.NotVersionedError(filename)
                interesting_files.add(filename)
                if tree.kind(filename) != 'directory':
                    continue
                for path, ie in tree.iter_entries_by_dir(specific_files=[filename]):
                    interesting_files.add(path)
            new_conflicts = conflicts.select_conflicts(tree, file_list)[0]
        else:
            allowed_conflicts = ('text conflict', 'contents conflict')
            restore_files = [c.path for c in conflicts if c.typestring in allowed_conflicts]
        _mod_merge.transform_tree(tree, tree.basis_tree(), interesting_files)
        tree.set_conflicts(new_conflicts)
        if file_list is not None:
            restore_files = file_list
        for filename in restore_files:
            try:
                restore(tree.abspath(filename))
            except errors.NotConflicted:
                pass
        tree.set_parent_ids(parents[:1])
        try:
            merger = _mod_merge.Merger.from_revision_ids(tree, parents[1])
            merger.interesting_files = interesting_files
            merger.merge_type = merge_type
            merger.show_base = show_base
            merger.reprocess = reprocess
            conflicts = merger.do_merge()
        finally:
            tree.set_parent_ids(parents)
        if len(conflicts) > 0:
            return 1
        else:
            return 0