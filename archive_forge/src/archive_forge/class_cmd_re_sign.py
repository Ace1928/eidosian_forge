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
class cmd_re_sign(Command):
    __doc__ = 'Create a digital signature for an existing revision.'
    hidden = True
    takes_args = ['revision_id*']
    takes_options = ['directory', 'revision']

    def run(self, revision_id_list=None, revision=None, directory='.'):
        if revision_id_list is not None and revision is not None:
            raise errors.CommandError(gettext('You can only supply one of revision_id or --revision'))
        if revision_id_list is None and revision is None:
            raise errors.CommandError(gettext('You must supply either --revision or a revision_id'))
        b = WorkingTree.open_containing(directory)[0].branch
        self.enter_context(b.lock_write())
        return self._run(b, revision_id_list, revision)

    def _run(self, b, revision_id_list, revision):
        from .repository import WriteGroup
        gpg_strategy = gpg.GPGStrategy(b.get_config_stack())
        if revision_id_list is not None:
            with WriteGroup(b.repository):
                for revision_id in revision_id_list:
                    revision_id = revision_id.encode('utf-8')
                    b.repository.sign_revision(revision_id, gpg_strategy)
        elif revision is not None:
            if len(revision) == 1:
                revno, rev_id = revision[0].in_history(b)
                with WriteGroup(b.repository):
                    b.repository.sign_revision(rev_id, gpg_strategy)
            elif len(revision) == 2:
                from_revno, from_revid = revision[0].in_history(b)
                to_revno, to_revid = revision[1].in_history(b)
                if to_revid is None:
                    to_revno = b.revno()
                if from_revno is None or to_revno is None:
                    raise errors.CommandError(gettext('Cannot sign a range of non-revision-history revisions'))
                with WriteGroup(b.repository):
                    for revno in range(from_revno, to_revno + 1):
                        b.repository.sign_revision(b.get_rev_id(revno), gpg_strategy)
            else:
                raise errors.CommandError(gettext('Please supply either one revision, or a range.'))