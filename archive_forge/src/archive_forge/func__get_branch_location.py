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
def _get_branch_location(control_dir, possible_transports=None):
    """Return location of branch for this control dir."""
    try:
        target = control_dir.get_branch_reference()
    except errors.NotBranchError:
        return control_dir.root_transport.base
    if target is not None:
        return target
    this_branch = control_dir.open_branch(possible_transports=possible_transports)
    master_location = this_branch.get_bound_location()
    if master_location is not None:
        return master_location
    return this_branch.base