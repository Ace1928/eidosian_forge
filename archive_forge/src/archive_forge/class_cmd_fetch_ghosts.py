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
class cmd_fetch_ghosts(Command):
    __doc__ = 'Attempt to retrieve ghosts from another branch.\n\n    If the other branch is not supplied, the last-pulled branch is used.\n    '
    hidden = True
    aliases = ['fetch-missing']
    takes_args = ['branch?']
    takes_options = [Option('no-fix', help='Skip additional synchonization.')]

    def run(self, branch=None, no_fix=False):
        from .fetch_ghosts import GhostFetcher
        installed, failed = GhostFetcher.from_cmdline(branch).run()
        if len(installed) > 0:
            self.outf.write('Installed:\n')
            for rev in installed:
                self.outf.write(rev.decode('utf-8') + '\n')
        if len(failed) > 0:
            self.outf.write('Still missing:\n')
            for rev in failed:
                self.outf.write(rev.decode('utf-8') + '\n')
        if not no_fix and len(installed) > 0:
            cmd_reconcile().run('.')