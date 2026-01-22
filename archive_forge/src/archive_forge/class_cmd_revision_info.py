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
class cmd_revision_info(Command):
    __doc__ = 'Show revision number and revision id for a given revision identifier.\n    '
    hidden = True
    takes_args = ['revision_info*']
    takes_options = ['revision', custom_help('directory', help='Branch to examine, rather than the one containing the working directory.'), Option('tree', help='Show revno of working tree.')]

    @display_command
    def run(self, revision=None, directory='.', tree=False, revision_info_list=[]):
        try:
            wt = WorkingTree.open_containing(directory)[0]
            b = wt.branch
            self.enter_context(wt.lock_read())
        except (errors.NoWorkingTree, errors.NotLocalUrl):
            wt = None
            b = Branch.open_containing(directory)[0]
            self.enter_context(b.lock_read())
        revision_ids = []
        if revision is not None:
            revision_ids.extend((rev.as_revision_id(b) for rev in revision))
        if revision_info_list is not None:
            for rev_str in revision_info_list:
                rev_spec = RevisionSpec.from_string(rev_str)
                revision_ids.append(rev_spec.as_revision_id(b))
        if len(revision_ids) == 0:
            if tree:
                if wt is None:
                    raise errors.NoWorkingTree(directory)
                revision_ids.append(wt.last_revision())
            else:
                revision_ids.append(b.last_revision())
        revinfos = []
        maxlen = 0
        for revision_id in revision_ids:
            try:
                dotted_revno = b.revision_id_to_dotted_revno(revision_id)
                revno = '.'.join((str(i) for i in dotted_revno))
            except errors.NoSuchRevision:
                revno = '???'
            maxlen = max(maxlen, len(revno))
            revinfos.append((revno, revision_id))
        self.cleanup_now()
        for revno, revid in revinfos:
            self.outf.write('%*s %s\n' % (maxlen, revno, revid.decode('utf-8')))