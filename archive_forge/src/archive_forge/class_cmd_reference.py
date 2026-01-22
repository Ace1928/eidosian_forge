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
class cmd_reference(Command):
    __doc__ = 'list, view and set branch locations for nested trees.\n\n    If no arguments are provided, lists the branch locations for nested trees.\n    If one argument is provided, display the branch location for that tree.\n    If two arguments are provided, set the branch location for that tree.\n    '
    hidden = True
    takes_args = ['path?', 'location?']
    takes_options = ['directory', Option('force-unversioned', help='Set reference even if path is not versioned.')]

    def run(self, path=None, directory='.', location=None, force_unversioned=False):
        tree, branch, relpath = controldir.ControlDir.open_containing_tree_or_branch(directory)
        if tree is None:
            tree = branch.basis_tree()
        if path is None:
            with tree.lock_read():
                info = [(path, tree.get_reference_info(path, branch)) for path in tree.iter_references()]
                self._display_reference_info(tree, branch, info)
        else:
            if not tree.is_versioned(path) and (not force_unversioned):
                raise errors.NotVersionedError(path)
            if location is None:
                info = [(path, tree.get_reference_info(path, branch))]
                self._display_reference_info(tree, branch, info)
            else:
                tree.set_reference_info(path, location)

    def _display_reference_info(self, tree, branch, info):
        ref_list = []
        for path, location in info:
            ref_list.append((path, location))
        for path, location in sorted(ref_list):
            self.outf.write('{} {}\n'.format(path, location))