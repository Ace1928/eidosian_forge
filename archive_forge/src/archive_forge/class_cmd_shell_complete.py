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
class cmd_shell_complete(Command):
    __doc__ = "Show appropriate completions for context.\n\n    For a list of all available commands, say 'brz shell-complete'.\n    "
    takes_args = ['context?']
    aliases = ['s-c']
    hidden = True

    @display_command
    def run(self, context=None):
        from . import shellcomplete
        shellcomplete.shellcomplete(context)