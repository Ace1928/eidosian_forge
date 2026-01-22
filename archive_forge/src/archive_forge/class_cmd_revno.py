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
class cmd_revno(Command):
    __doc__ = 'Show current revision number.\n\n    This is equal to the number of revisions on this branch.\n    '
    _see_also = ['info']
    takes_args = ['location?']
    takes_options = [Option('tree', help='Show revno of working tree.'), 'revision']

    @display_command
    def run(self, tree=False, location='.', revision=None):
        if revision is not None and tree:
            raise errors.CommandError(gettext('--tree and --revision can not be used together'))
        if tree:
            try:
                wt = WorkingTree.open_containing(location)[0]
                self.enter_context(wt.lock_read())
            except (errors.NoWorkingTree, errors.NotLocalUrl) as exc:
                raise errors.NoWorkingTree(location) from exc
            b = wt.branch
            revid = wt.last_revision()
        else:
            b = Branch.open_containing(location)[0]
            self.enter_context(b.lock_read())
            if revision:
                if len(revision) != 1:
                    raise errors.CommandError(gettext('Revision numbers only make sense for single revisions, not ranges'))
                revid = revision[0].as_revision_id(b)
            else:
                revid = b.last_revision()
        try:
            revno_t = b.revision_id_to_dotted_revno(revid)
        except (errors.NoSuchRevision, errors.GhostRevisionsHaveNoRevno):
            revno_t = ('???',)
        revno = '.'.join((str(n) for n in revno_t))
        self.cleanup_now()
        self.outf.write(revno + '\n')