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
class cmd_plugins(Command):
    __doc__ = 'List the installed plugins.\n\n    This command displays the list of installed plugins including\n    version of plugin and a short description of each.\n\n    --verbose shows the path where each plugin is located.\n\n    A plugin is an external component for Breezy that extends the\n    revision control system, by adding or replacing code in Breezy.\n    Plugins can do a variety of things, including overriding commands,\n    adding new commands, providing additional network transports and\n    customizing log output.\n\n    See the Breezy Plugin Guide <https://www.breezy-vcs.org/doc/plugins/en/>\n    for further information on plugins including where to find them and how to\n    install them. Instructions are also provided there on how to write new\n    plugins using the Python programming language.\n    '
    takes_options = ['verbose']

    @display_command
    def run(self, verbose=False):
        from . import plugin
        self.outf.writelines(list(plugin.describe_plugins(show_paths=verbose)))