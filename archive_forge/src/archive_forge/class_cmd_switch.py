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
class cmd_switch(Command):
    __doc__ = "Set the branch of a checkout and update.\n\n    For lightweight checkouts, this changes the branch being referenced.\n    For heavyweight checkouts, this checks that there are no local commits\n    versus the current bound branch, then it makes the local branch a mirror\n    of the new location and binds to it.\n\n    In both cases, the working tree is updated and uncommitted changes\n    are merged. The user can commit or revert these as they desire.\n\n    Pending merges need to be committed or reverted before using switch.\n\n    The path to the branch to switch to can be specified relative to the parent\n    directory of the current branch. For example, if you are currently in a\n    checkout of /path/to/branch, specifying 'newbranch' will find a branch at\n    /path/to/newbranch.\n\n    Bound branches use the nickname of its master branch unless it is set\n    locally, in which case switching will update the local nickname to be\n    that of the master.\n    "
    takes_args = ['to_location?']
    takes_options = ['directory', Option('force', help='Switch even if local commits will be lost.'), 'revision', Option('create-branch', short_name='b', help='Create the target branch from this one before switching to it.'), Option('store', help='Store and restore uncommitted changes in the branch.')]

    def run(self, to_location=None, force=False, create_branch=False, revision=None, directory='.', store=False):
        from . import switch
        tree_location = directory
        revision = _get_one_revision('switch', revision)
        control_dir = controldir.ControlDir.open_containing(tree_location)[0]
        possible_transports = [control_dir.root_transport]
        if to_location is None:
            if revision is None:
                raise errors.CommandError(gettext('You must supply either a revision or a location'))
            to_location = tree_location
        try:
            branch = control_dir.open_branch(possible_transports=possible_transports)
            had_explicit_nick = branch.get_config().has_explicit_nickname()
        except errors.NotBranchError:
            branch = None
            had_explicit_nick = False
        else:
            possible_transports.append(branch.user_transport)
        if create_branch:
            if branch is None:
                raise errors.CommandError(gettext('cannot create branch without source branch'))
            to_location = lookup_new_sibling_branch(control_dir, to_location, possible_transports=possible_transports)
            if revision is not None:
                revision = revision.as_revision_id(branch)
            to_branch = branch.controldir.sprout(to_location, possible_transports=possible_transports, revision_id=revision, source_branch=branch).open_branch()
        else:
            try:
                to_branch = Branch.open(to_location, possible_transports=possible_transports)
            except errors.NotBranchError:
                to_branch = open_sibling_branch(control_dir, to_location, possible_transports=possible_transports)
            if revision is not None:
                revision = revision.as_revision_id(to_branch)
        possible_transports.append(to_branch.user_transport)
        try:
            switch.switch(control_dir, to_branch, force, revision_id=revision, store_uncommitted=store, possible_transports=possible_transports)
        except controldir.BranchReferenceLoop as exc:
            raise errors.CommandError(gettext('switching would create a branch reference loop. Use the "bzr up" command to switch to a different revision.')) from exc
        if had_explicit_nick:
            branch = control_dir.open_branch()
            branch.nick = to_branch.nick
        if to_branch.name:
            if to_branch.controldir.control_url != control_dir.control_url:
                note(gettext('Switched to branch %s at %s'), to_branch.name, urlutils.unescape_for_display(to_branch.base, 'utf-8'))
            else:
                note(gettext('Switched to branch %s'), to_branch.name)
        else:
            note(gettext('Switched to branch at %s'), urlutils.unescape_for_display(to_branch.base, 'utf-8'))