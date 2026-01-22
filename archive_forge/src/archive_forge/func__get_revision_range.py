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
def _get_revision_range(revisionspec_list, branch, command_name):
    """Take the input of a revision option and turn it into a revision range.

    It returns RevisionInfo objects which can be used to obtain the rev_id's
    of the desired revisions. It does some user input validations.
    """
    if revisionspec_list is None:
        rev1 = None
        rev2 = None
    elif len(revisionspec_list) == 1:
        rev1 = rev2 = revisionspec_list[0].in_history(branch)
    elif len(revisionspec_list) == 2:
        start_spec = revisionspec_list[0]
        end_spec = revisionspec_list[1]
        if end_spec.get_branch() != start_spec.get_branch():
            raise errors.CommandError(gettext("brz %s doesn't accept two revisions in different branches.") % command_name)
        if start_spec.spec is None:
            rev1 = RevisionInfo(branch, None, None)
        else:
            rev1 = start_spec.in_history(branch)
        if end_spec.spec is None:
            last_revno, last_revision_id = branch.last_revision_info()
            rev2 = RevisionInfo(branch, last_revno, last_revision_id)
        else:
            rev2 = end_spec.in_history(branch)
    else:
        raise errors.CommandError(gettext('brz %s --revision takes one or two values.') % command_name)
    return (rev1, rev2)