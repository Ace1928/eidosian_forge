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
class cmd_find_merge_base(Command):
    __doc__ = 'Find and print a base revision for merging two branches.'
    takes_args = ['branch', 'other']
    hidden = True

    @display_command
    def run(self, branch, other):
        branch1 = Branch.open_containing(branch)[0]
        branch2 = Branch.open_containing(other)[0]
        self.enter_context(branch1.lock_read())
        self.enter_context(branch2.lock_read())
        last1 = branch1.last_revision()
        last2 = branch2.last_revision()
        graph = branch1.repository.get_graph(branch2.repository)
        base_rev_id = graph.find_unique_lca(last1, last2)
        self.outf.write(gettext('merge base is revision %s\n') % base_rev_id.decode('utf-8'))