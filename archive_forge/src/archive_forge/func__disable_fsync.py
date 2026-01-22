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
def _disable_fsync(self):
    """Change the 'os' functionality to not synchronize."""
    self._orig_fsync = getattr(os, 'fsync', None)
    if self._orig_fsync is not None:
        os.fsync = lambda filedes: None
    self._orig_fdatasync = getattr(os, 'fdatasync', None)
    if self._orig_fdatasync is not None:
        os.fdatasync = lambda filedes: None