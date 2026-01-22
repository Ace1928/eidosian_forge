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
class cmd_tags(Command):
    __doc__ = 'List tags.\n\n    This command shows a table of tag names and the revisions they reference.\n    '
    _see_also = ['tag']
    takes_options = [custom_help('directory', help='Branch whose tags should be displayed.'), RegistryOption('sort', 'Sort tags by different criteria.', title='Sorting', lazy_registry=('breezy.tag', 'tag_sort_methods')), 'show-ids', 'revision']

    @display_command
    def run(self, directory='.', sort=None, show_ids=False, revision=None):
        from .tag import tag_sort_methods
        branch, relpath = Branch.open_containing(directory)
        tags = list(branch.tags.get_tag_dict().items())
        if not tags:
            return
        self.enter_context(branch.lock_read())
        if revision:
            tags = self._tags_for_range(branch, revision)
        if sort is None:
            sort = tag_sort_methods.get()
        sort(branch, tags)
        if not show_ids:
            for index, (tag, revid) in enumerate(tags):
                try:
                    revno = branch.revision_id_to_dotted_revno(revid)
                    if isinstance(revno, tuple):
                        revno = '.'.join(map(str, revno))
                except (errors.NoSuchRevision, errors.GhostRevisionsHaveNoRevno, errors.UnsupportedOperation):
                    revno = '?'
                tags[index] = (tag, revno)
        else:
            tags = [(tag, revid.decode('utf-8')) for tag, revid in tags]
        self.cleanup_now()
        for tag, revspec in tags:
            self.outf.write('%-20s %s\n' % (tag, revspec))

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