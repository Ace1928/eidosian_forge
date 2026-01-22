from .. import ui
from ..branch import Branch
from ..check import Check
from ..i18n import gettext
from ..revision import NULL_REVISION
from ..trace import note
from ..workingtree import WorkingTree
def _check_one_rev(self, rev_id, rev):
    """Cross-check one revision.

        :param rev_id: A revision id to check.
        :param rev: A revision or None to indicate a missing revision.
        """
    if rev.revision_id != rev_id:
        self._report_items.append(gettext('Mismatched internal revid {{{0}}} and index revid {{{1}}}').format(rev.revision_id.decode('utf-8'), rev_id.decode('utf-8')))
        rev_id = rev.revision_id
    self.planned_revisions.add(rev_id)
    self.ghosts.discard(rev_id)
    for parent in rev.parent_ids:
        if parent not in self.planned_revisions:
            self.ghosts.add(parent)
    self.ancestors[rev_id] = tuple(rev.parent_ids) or (NULL_REVISION,)
    self.add_pending_item(rev_id, ('inventories', rev_id), 'inventory', rev.inventory_sha1)
    self.checked_rev_cnt += 1