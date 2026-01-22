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
class cmd_ignored(Command):
    __doc__ = 'List ignored files and the patterns that matched them.\n\n    List all the ignored files and the ignore pattern that caused the file to\n    be ignored.\n\n    Alternatively, to list just the files::\n\n        brz ls --ignored\n    '
    encoding_type = 'replace'
    _see_also = ['ignore', 'ls']
    takes_options = ['directory']

    @display_command
    def run(self, directory='.'):
        tree = WorkingTree.open_containing(directory)[0]
        self.enter_context(tree.lock_read())
        for path, file_class, kind, entry in tree.list_files():
            if file_class != 'I':
                continue
            pat = tree.is_ignored(path)
            self.outf.write('%-50s %s\n' % (path, pat))