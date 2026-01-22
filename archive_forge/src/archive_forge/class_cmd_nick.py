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
class cmd_nick(Command):
    __doc__ = 'Print or set the branch nickname.\n\n    If unset, the colocated branch name is used for colocated branches, and\n    the branch directory name is used for other branches.  To print the\n    current nickname, execute with no argument.\n\n    Bound branches use the nickname of its master branch unless it is set\n    locally.\n    '
    _see_also = ['info']
    takes_args = ['nickname?']
    takes_options = ['directory']

    def run(self, nickname=None, directory='.'):
        branch = Branch.open_containing(directory)[0]
        if nickname is None:
            self.printme(branch)
        else:
            branch.nick = nickname

    @display_command
    def printme(self, branch):
        self.outf.write('%s\n' % branch.nick)