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
class cmd_unshelve(Command):
    __doc__ = "Restore shelved changes.\n\n    By default, the most recently shelved changes are restored. However if you\n    specify a shelf by id those changes will be restored instead.  This works\n    best when the changes don't depend on each other.\n    "
    takes_args = ['shelf_id?']
    takes_options = ['directory', RegistryOption.from_kwargs('action', help='The action to perform.', enum_switch=False, value_switches=True, apply='Apply changes and remove from the shelf.', dry_run='Show changes, but do not apply or remove them.', preview='Instead of unshelving the changes, show the diff that would result from unshelving.', delete_only='Delete changes without applying them.', keep="Apply changes but don't delete them.")]
    _see_also = ['shelve']

    def run(self, shelf_id=None, action='apply', directory='.'):
        from .shelf_ui import Unshelver
        unshelver = Unshelver.from_args(shelf_id, action, directory=directory)
        try:
            unshelver.run()
        finally:
            unshelver.tree.unlock()