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
def _get_merger_from_branch(self, tree, location, revision, remember, possible_transports, pb):
    """Produce a merger from a location, assuming it refers to a branch."""
    other_loc, user_location = self._select_branch_location(tree, location, revision, -1)
    if revision is not None and len(revision) == 2:
        base_loc, _unused = self._select_branch_location(tree, location, revision, 0)
    else:
        base_loc = other_loc
    other_branch, other_path = Branch.open_containing(other_loc, possible_transports)
    if base_loc == other_loc:
        base_branch = other_branch
    else:
        base_branch, base_path = Branch.open_containing(base_loc, possible_transports)
    other_revision_id = None
    base_revision_id = None
    if revision is not None:
        if len(revision) >= 1:
            other_revision_id = revision[-1].as_revision_id(other_branch)
        if len(revision) == 2:
            base_revision_id = revision[0].as_revision_id(base_branch)
    if other_revision_id is None:
        other_revision_id = other_branch.last_revision()
    if user_location is not None and (remember or (remember is None and tree.branch.get_submit_branch() is None)):
        tree.branch.set_submit_branch(other_branch.base)
    other_branch.tags.merge_to(tree.branch.tags, ignore_master=True)
    merger = _mod_merge.Merger.from_revision_ids(tree, other_revision_id, base_revision_id, other_branch, base_branch)
    if other_path != '':
        allow_pending = False
        merger.interesting_files = [other_path]
    else:
        allow_pending = True
    return (merger, allow_pending)