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
class cmd_reconfigure(Command):
    __doc__ = 'Reconfigure the type of a brz directory.\n\n    A target configuration must be specified.\n\n    For checkouts, the bind-to location will be auto-detected if not specified.\n    The order of preference is\n    1. For a lightweight checkout, the current bound location.\n    2. For branches that used to be checkouts, the previously-bound location.\n    3. The push location.\n    4. The parent location.\n    If none of these is available, --bind-to must be specified.\n    '
    _see_also = ['branches', 'checkouts', 'standalone-trees', 'working-trees']
    takes_args = ['location?']
    takes_options = [RegistryOption.from_kwargs('tree_type', title='Tree type', help='The relation between branch and tree.', value_switches=True, enum_switch=False, branch='Reconfigure to be an unbound branch with no working tree.', tree='Reconfigure to be an unbound branch with a working tree.', checkout='Reconfigure to be a bound branch with a working tree.', lightweight_checkout='Reconfigure to be a lightweight checkout (with no local history).'), RegistryOption.from_kwargs('repository_type', title='Repository type', help='Location fo the repository.', value_switches=True, enum_switch=False, standalone='Reconfigure to be a standalone branch (i.e. stop using shared repository).', use_shared='Reconfigure to use a shared repository.'), RegistryOption.from_kwargs('repository_trees', title='Trees in Repository', help='Whether new branches in the repository have trees.', value_switches=True, enum_switch=False, with_trees='Reconfigure repository to create working trees on branches by default.', with_no_trees='Reconfigure repository to not create working trees on branches by default.'), Option('bind-to', help='Branch to bind checkout to.', type=str), Option('force', help='Perform reconfiguration even if local changes will be lost.'), Option('stacked-on', help='Reconfigure a branch to be stacked on another branch.', type=str), Option('unstacked', help='Reconfigure a branch to be unstacked.  This may require copying substantial data into it.')]

    def run(self, location=None, bind_to=None, force=False, tree_type=None, repository_type=None, repository_trees=None, stacked_on=None, unstacked=None):
        directory = controldir.ControlDir.open(location)
        if stacked_on and unstacked:
            raise errors.CommandError(gettext("Can't use both --stacked-on and --unstacked"))
        elif stacked_on is not None:
            reconfigure.ReconfigureStackedOn().apply(directory, stacked_on)
        elif unstacked:
            reconfigure.ReconfigureUnstacked().apply(directory)
        if tree_type is None and repository_type is None and (repository_trees is None):
            if stacked_on or unstacked:
                return
            else:
                raise errors.CommandError(gettext('No target configuration specified'))
        reconfiguration = None
        if tree_type == 'branch':
            reconfiguration = reconfigure.Reconfigure.to_branch(directory)
        elif tree_type == 'tree':
            reconfiguration = reconfigure.Reconfigure.to_tree(directory)
        elif tree_type == 'checkout':
            reconfiguration = reconfigure.Reconfigure.to_checkout(directory, bind_to)
        elif tree_type == 'lightweight-checkout':
            reconfiguration = reconfigure.Reconfigure.to_lightweight_checkout(directory, bind_to)
        if reconfiguration:
            reconfiguration.apply(force)
            reconfiguration = None
        if repository_type == 'use-shared':
            reconfiguration = reconfigure.Reconfigure.to_use_shared(directory)
        elif repository_type == 'standalone':
            reconfiguration = reconfigure.Reconfigure.to_standalone(directory)
        if reconfiguration:
            reconfiguration.apply(force)
            reconfiguration = None
        if repository_trees == 'with-trees':
            reconfiguration = reconfigure.Reconfigure.set_repository_trees(directory, True)
        elif repository_trees == 'with-no-trees':
            reconfiguration = reconfigure.Reconfigure.set_repository_trees(directory, False)
        if reconfiguration:
            reconfiguration.apply(force)
            reconfiguration = None