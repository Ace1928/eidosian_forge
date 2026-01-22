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
def _do_merge(self, merger, change_reporter, allow_pending, verified):
    merger.change_reporter = change_reporter
    conflict_count = len(merger.do_merge())
    if allow_pending:
        merger.set_pending()
    if verified == 'failed':
        warning('Preview patch does not match changes')
    if conflict_count != 0:
        return 1
    else:
        return 0