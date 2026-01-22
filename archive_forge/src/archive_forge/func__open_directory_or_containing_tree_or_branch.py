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
def _open_directory_or_containing_tree_or_branch(filename, directory):
    """Open the tree or branch containing the specified file, unless
    the --directory option is used to specify a different branch."""
    if directory is not None:
        return (None, Branch.open(directory), filename)
    return controldir.ControlDir.open_containing_tree_or_branch(filename)