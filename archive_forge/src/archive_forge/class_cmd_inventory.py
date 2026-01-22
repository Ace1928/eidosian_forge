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
class cmd_inventory(Command):
    __doc__ = 'Show inventory of the current working copy or a revision.\n\n    It is possible to limit the output to a particular entry\n    type using the --kind option.  For example: --kind file.\n\n    It is also possible to restrict the list of files to a specific\n    set. For example: brz inventory --show-ids this/file\n    '
    hidden = True
    _see_also = ['ls']
    takes_options = ['revision', 'show-ids', Option('include-root', help='Include the entry for the root of the tree, if any.'), Option('kind', help='List entries of a particular kind: file, directory, symlink.', type=str)]
    takes_args = ['file*']

    @display_command
    def run(self, revision=None, show_ids=False, kind=None, include_root=False, file_list=None):
        if kind and kind not in ['file', 'directory', 'symlink']:
            raise errors.CommandError(gettext('invalid kind %r specified') % (kind,))
        revision = _get_one_revision('inventory', revision)
        work_tree, file_list = WorkingTree.open_containing_paths(file_list)
        self.enter_context(work_tree.lock_read())
        if revision is not None:
            tree = revision.as_tree(work_tree.branch)
            extra_trees = [work_tree]
            self.enter_context(tree.lock_read())
        else:
            tree = work_tree
            extra_trees = []
        self.enter_context(tree.lock_read())
        if file_list is not None:
            paths = tree.find_related_paths_across_trees(file_list, extra_trees, require_versioned=True)
            entries = tree.iter_entries_by_dir(specific_files=paths)
        else:
            entries = tree.iter_entries_by_dir()
        for path, entry in sorted(entries):
            if kind and kind != entry.kind:
                continue
            if path == '' and (not include_root):
                continue
            if show_ids:
                self.outf.write('%-50s %s\n' % (path, entry.file_id.decode('utf-8')))
            else:
                self.outf.write(path)
                self.outf.write('\n')