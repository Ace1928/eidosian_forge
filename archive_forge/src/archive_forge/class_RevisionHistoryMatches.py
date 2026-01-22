from testtools.matchers import Equals, Matcher, Mismatch
from .. import osutils
from .. import revision as _mod_revision
from ..tree import InterTree, TreeChange
class RevisionHistoryMatches(Matcher):
    """A matcher that checks if a branch has a specific revision history.

    :ivar history: Revision history, as list of revisions. Oldest first.
    """

    def __init__(self, history):
        Matcher.__init__(self)
        self.expected = history

    def __str__(self):
        return 'RevisionHistoryMatches(%r)' % self.expected

    def match(self, branch):
        with branch.lock_read():
            graph = branch.repository.get_graph()
            history = list(graph.iter_lefthand_ancestry(branch.last_revision(), [_mod_revision.NULL_REVISION]))
            history.reverse()
        return Equals(self.expected).match(history)