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
class cmd_mkdir(Command):
    __doc__ = 'Create a new versioned directory.\n\n    This is equivalent to creating the directory and then adding it.\n    '
    takes_args = ['dir+']
    takes_options = [Option('parents', help='No error if existing, make parent directories as needed.', short_name='p')]
    encoding_type = 'replace'

    @classmethod
    def add_file_with_parents(cls, wt, relpath):
        if wt.is_versioned(relpath):
            return
        cls.add_file_with_parents(wt, osutils.dirname(relpath))
        wt.add([relpath])

    @classmethod
    def add_file_single(cls, wt, relpath):
        wt.add([relpath])

    def run(self, dir_list, parents=False):
        if parents:
            add_file = self.add_file_with_parents
        else:
            add_file = self.add_file_single
        for dir in dir_list:
            wt, relpath = WorkingTree.open_containing(dir)
            if parents:
                try:
                    os.makedirs(dir)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
            else:
                os.mkdir(dir)
            add_file(wt, relpath)
            if not is_quiet():
                self.outf.write(gettext('added %s\n') % dir)