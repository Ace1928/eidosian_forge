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
def _tags_for_range(self, branch, revision):
    rev1, rev2 = _get_revision_range(revision, branch, self.name())
    revid1, revid2 = (rev1.rev_id, rev2.rev_id)
    if revid1 and revid1 != revid2:
        if branch.repository.get_graph().is_ancestor(revid2, revid1):
            return []
    tagged_revids = branch.tags.get_reverse_tag_dict()
    found = []
    for r in branch.iter_merge_sorted_revisions(start_revision_id=revid2, stop_revision_id=revid1, stop_rule='include'):
        revid_tags = tagged_revids.get(r[0], None)
        if revid_tags:
            found.extend([(tag, r[0]) for tag in revid_tags])
    return found