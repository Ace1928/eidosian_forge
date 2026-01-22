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
class cmd_export_pot(Command):
    __doc__ = 'Export command helps and error messages in po format.'
    hidden = True
    takes_options = [Option('plugin', help='Export help text from named command (defaults to all built in commands).', type=str), Option('include-duplicates', help='Output multiple copies of the same msgid string if it appears more than once.')]

    def run(self, plugin=None, include_duplicates=False):
        from .export_pot import export_pot
        export_pot(self.outf, plugin, include_duplicates)