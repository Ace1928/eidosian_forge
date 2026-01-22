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
class cmd_patch(Command):
    """Apply a named patch to the current tree.

    """
    takes_args = ['filename?']
    takes_options = [Option('strip', type=int, short_name='p', help='Strip the smallest prefix containing num leading slashes from filenames.'), Option('silent', help='Suppress chatter.')]

    def run(self, filename=None, strip=None, silent=False):
        from .patch import patch_tree
        wt = WorkingTree.open_containing('.')[0]
        if strip is None:
            strip = 1
        my_file = None
        if filename is None:
            my_file = getattr(sys.stdin, 'buffer', sys.stdin)
        else:
            my_file = open(filename, 'rb')
        patches = [my_file.read()]
        return patch_tree(wt, patches, strip, quiet=is_quiet(), out=self.outf)