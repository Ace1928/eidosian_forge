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
class cmd_join(Command):
    __doc__ = 'Combine a tree into its containing tree.\n\n    This command requires the target tree to be in a rich-root format.\n\n    The TREE argument should be an independent tree, inside another tree, but\n    not part of it.  (Such trees can be produced by "brz split", but also by\n    running "brz branch" with the target inside a tree.)\n\n    The result is a combined tree, with the subtree no longer an independent\n    part.  This is marked as a merge of the subtree into the containing tree,\n    and all history is preserved.\n    '
    _see_also = ['split']
    takes_args = ['tree']
    takes_options = [Option('reference', help='Join by reference.', hidden=True)]

    def run(self, tree, reference=False):
        from breezy.mutabletree import BadReferenceTarget
        sub_tree = WorkingTree.open(tree)
        parent_dir = osutils.dirname(sub_tree.basedir)
        containing_tree = WorkingTree.open_containing(parent_dir)[0]
        repo = containing_tree.branch.repository
        if not repo.supports_rich_root():
            raise errors.CommandError(gettext("Can't join trees because %s doesn't support rich root data.\nYou can use brz upgrade on the repository.") % (repo,))
        if reference:
            try:
                containing_tree.add_reference(sub_tree)
            except BadReferenceTarget as exc:
                raise errors.CommandError(gettext('Cannot join {0}.  {1}').format(tree, exc.reason)) from exc
        else:
            try:
                containing_tree.subsume(sub_tree)
            except errors.BadSubsumeSource as exc:
                raise errors.CommandError(gettext('Cannot join {0}.  {1}').format(tree, exc.reason)) from exc