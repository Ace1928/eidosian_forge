import sys
from . import revision as _mod_revision
from .commands import Command
from .controldir import ControlDir
from .errors import CommandError
from .option import Option
from .trace import note
def _find_range_and_middle(self, branch_last_rev=None):
    """Find the current revision range, and the midpoint."""
    self._load_tree()
    self._middle_revid = None
    if not branch_last_rev:
        last_revid = self._branch.last_revision()
    else:
        last_revid = branch_last_rev
    repo = self._branch.repository
    with repo.lock_read():
        graph = repo.get_graph()
        rev_sequence = graph.iter_lefthand_ancestry(last_revid, (_mod_revision.NULL_REVISION,))
        high_revid = None
        low_revid = None
        between_revs = []
        for revision in rev_sequence:
            between_revs.insert(0, revision)
            matches = [x[1] for x in self._items if x[0] == revision and x[1] in ('yes', 'no')]
            if not matches:
                continue
            if len(matches) > 1:
                raise RuntimeError('revision %s duplicated' % revision)
            if matches[0] == 'yes':
                high_revid = revision
                between_revs = []
            elif matches[0] == 'no':
                low_revid = revision
                del between_revs[0]
                break
        if not high_revid:
            high_revid = last_revid
        if not low_revid:
            low_revid = self._branch.get_rev_id(1)
    spread = len(between_revs) + 1
    if spread < 2:
        middle_index = 0
    else:
        middle_index = spread // 2 - 1
    if len(between_revs) > 0:
        self._middle_revid = between_revs[middle_index]
    else:
        self._middle_revid = high_revid
    self._high_revid = high_revid
    self._low_revid = low_revid