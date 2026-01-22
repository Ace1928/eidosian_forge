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
class cmd_bind(Command):
    __doc__ = 'Convert the current branch into a checkout of the supplied branch.\n    If no branch is supplied, rebind to the last bound location.\n\n    Once converted into a checkout, commits must succeed on the master branch\n    before they will be applied to the local branch.\n\n    Bound branches use the nickname of its master branch unless it is set\n    locally, in which case binding will update the local nickname to be\n    that of the master.\n    '
    _see_also = ['checkouts', 'unbind']
    takes_args = ['location?']
    takes_options = ['directory']

    def run(self, location=None, directory='.'):
        b, relpath = Branch.open_containing(directory)
        if location is None:
            try:
                location = b.get_old_bound_location()
            except errors.UpgradeRequired as exc:
                raise errors.CommandError(gettext('No location supplied.  This format does not remember old locations.')) from exc
            else:
                if location is None:
                    if b.get_bound_location() is not None:
                        raise errors.CommandError(gettext('Branch is already bound'))
                    else:
                        raise errors.CommandError(gettext('No location supplied and no previous location known'))
        b_other = Branch.open(location)
        try:
            b.bind(b_other)
        except errors.DivergedBranches as exc:
            raise errors.CommandError(gettext('These branches have diverged. Try merging, and then bind again.')) from exc
        if b.get_config().has_explicit_nickname():
            b.nick = b_other.nick