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
class cmd_import(Command):
    __doc__ = 'Import sources from a directory, tarball or zip file\n\n    This command will import a directory, tarball or zip file into a bzr\n    tree, replacing any versioned files already present.  If a directory is\n    specified, it is used as the target.  If the directory does not exist, or\n    is not versioned, it is created.\n\n    Tarballs may be gzip or bzip2 compressed.  This is autodetected.\n\n    If the tarball or zip has a single root directory, that directory is\n    stripped when extracting the tarball.  This is not done for directories.\n    '
    takes_args = ['source', 'tree?']

    def run(self, source, tree=None):
        from .upstream_import import do_import
        do_import(source, tree)