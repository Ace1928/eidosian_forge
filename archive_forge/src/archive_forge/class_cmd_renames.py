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
class cmd_renames(Command):
    __doc__ = 'Show list of renamed files.\n    '
    _see_also = ['status']
    takes_args = ['dir?']

    @display_command
    def run(self, dir='.'):
        tree = WorkingTree.open_containing(dir)[0]
        self.enter_context(tree.lock_read())
        old_tree = tree.basis_tree()
        self.enter_context(old_tree.lock_read())
        renames = []
        iterator = tree.iter_changes(old_tree, include_unchanged=True)
        for change in iterator:
            if change.path[0] == change.path[1]:
                continue
            if None in change.path:
                continue
            renames.append(change.path)
        renames.sort()
        for old_name, new_name in renames:
            self.outf.write('{} => {}\n'.format(old_name, new_name))